# chrome_whisper_transcriber_local.py
# -*- coding: utf-8 -*-
import os
import sys
import time
import re
import json
import unicodedata
import threading
import subprocess
import importlib
import datetime as dt
import inspect
import queue
import ctypes
import ast
import tempfile
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import requests



def _is_frozen_bundle() -> bool:
    return getattr(sys, "frozen", False)


def _app_base_dir() -> str:
    if _is_frozen_bundle():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_writable_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(path, ".perm_check")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
    finally:
        if os.path.exists(probe):
            os.remove(probe)
    return path


def _resolve_output_dir() -> str:
    override = os.environ.get("CWT_OUTPUT_DIR")
    if override:
        return _ensure_writable_dir(os.path.abspath(os.path.expanduser(override)))

    if _is_frozen_bundle():
        candidates = []
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            candidates.append(os.path.join(local_app, "ChromeWhisperTranscriber", "output"))
        home = os.path.expanduser("~")
        if home:
            candidates.append(os.path.join(home, "Documents", "ChromeWhisperTranscriber", "output"))
            candidates.append(os.path.join(home, "ChromeWhisperTranscriber", "output"))
        candidates.append(os.path.join(tempfile.gettempdir(), "ChromeWhisperTranscriber", "output"))
        for cand in candidates:
            try:
                return _ensure_writable_dir(cand)
            except Exception:
                continue
        raise RuntimeError("出力フォルダを作成できませんでした。CWT_OUTPUT_DIR 環境変数で保存先を指定してください。")

    return _ensure_writable_dir(os.path.join(_app_base_dir(), "output"))

# ========= 出力先 =========
BASE_DIR = _app_base_dir()
OUTPUT_DIR = _resolve_output_dir()


def _normalize_dir(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _resolve_initial_dir(env_name: str, fallback: str) -> str:
    candidate = os.environ.get(env_name)
    if candidate:
        try:
            return _ensure_writable_dir(_normalize_dir(candidate))
        except Exception as exc:
            print(f"[WARN] {env_name}={candidate} を利用できません: {exc}", file=sys.stderr)
    return _ensure_writable_dir(fallback)


AUDIO_OUTPUT_DIR = _resolve_initial_dir("CWT_AUDIO_DIR", OUTPUT_DIR)
SUMMARY_OUTPUT_DIR = _resolve_initial_dir("CWT_SUMMARY_DIR", OUTPUT_DIR)
TMP_DIR = _ensure_writable_dir(os.path.join(SUMMARY_OUTPUT_DIR, "_tmp"))
INDEX_MD = os.path.join(SUMMARY_OUTPUT_DIR, "_index.md")  # 一覧


def set_audio_output_dir(path: str) -> str:
    global AUDIO_OUTPUT_DIR
    AUDIO_OUTPUT_DIR = _ensure_writable_dir(_normalize_dir(path))
    return AUDIO_OUTPUT_DIR


def set_summary_output_dir(path: str) -> str:
    global SUMMARY_OUTPUT_DIR, TMP_DIR, INDEX_MD
    SUMMARY_OUTPUT_DIR = _ensure_writable_dir(_normalize_dir(path))
    TMP_DIR = _ensure_writable_dir(os.path.join(SUMMARY_OUTPUT_DIR, "_tmp"))
    INDEX_MD = os.path.join(SUMMARY_OUTPUT_DIR, "_index.md")
    return SUMMARY_OUTPUT_DIR

# ========= 科目リスト =========
SUBJECTS = [
    "自動判定",
    "数学","地理総合","歴史総合","公共","言語文化","現代の国語",
    "科学と人間生活","英語コミュニケーション","情報","家庭基礎","体育","その他"
]

# ========= Ollama（要約/科目判定/タイトル） =========
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# instruct系の方が遵守率高（pull済なら: llama3:8b-instruct）
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")

# ========= faster-whisper（文字起こし） =========
FW_MODEL = os.environ.get("FW_MODEL", "medium")       # "tiny"～"large-v3"
FW_DEVICE_CFG = os.environ.get("FW_DEVICE", "auto")   # "auto"|"cuda"|"cpu"
FW_COMPUTE_CFG = os.environ.get("FW_COMPUTE", "")     # 省略時は自動

# ========= 録音/ブロック =========
BLOCKSIZE    = int(os.environ.get("BLOCKSIZE", "2048"))      # sounddevice
SC_BLOCKSIZE = int(os.environ.get("SC_BLOCKSIZE","4096"))    # soundcard

# ========= 自動区切り（無音検知） =========
AUTO_CUT_ENABLED      = os.environ.get("AUTO_CUT_ENABLED", "1") != "0"
AUTO_CUT_SILENCE_SEC  = float(os.environ.get("AUTO_CUT_SILENCE_SEC", "5.0"))
AUTO_CUT_LEVEL_PCT    = float(os.environ.get("AUTO_CUT_LEVEL_PCT", "2.5"))
AUTO_CUT_MIN_SEC      = float(os.environ.get("AUTO_CUT_MIN_SEC", "1.0"))
AUTO_TRIM_PAD_SEC     = float(os.environ.get("AUTO_TRIM_PAD_SEC", "0.2"))

# ========= 要約の詳細度 =========
SUM_BULLETS_MAX = int(os.environ.get("SUM_BULLETS_MAX", "8"))
SUM_LINES_MAX   = int(os.environ.get("SUM_LINES_MAX", "6"))
SUM_TOKENS      = int(os.environ.get("SUM_TOKENS", "1400"))

# ========= 文字起こし整形パラメータ =========
PARA_GAP_SEC       = float(os.environ.get("PARA_GAP_SEC", "1.2"))  # 段落分けの無音閾値
SENTENCE_FALLBACK  = int(os.environ.get("SENTENCE_FALLBACK", "60"))# 句点が少ない時の分割しきい値

# ========= 状態 =========
@dataclass
class RecorderState:
    running: bool = False
    backend: str = "sd"  # "sd" | "sc"
    current_segment_frames: List[np.ndarray] = field(default_factory=list)  # [int16 1D]
    last_rms: float = 0.0
    stream: Optional[sd.RawInputStream] = None
    device_index: Optional[int] = None
    samplerate: Optional[int] = None
    channels: int = 2
    lock: threading.Lock = field(default_factory=threading.Lock)

    # soundcard用
    sc_thread: Optional[threading.Thread] = None
    sc_stop: threading.Event = field(default_factory=threading.Event)
    sc_mic_name: Optional[str] = None

    # 無音監視
    silence_run_sec: float = 0.0
    last_cut_at: float = 0.0

    def reset_segment(self):
        with self.lock:
            self.current_segment_frames = []
            self.silence_run_sec = 0.0

state = RecorderState()

# キュー（手動/自動区切りの順次処理）
@dataclass
class Job:
    wav_path: str
    chosen_subject: Optional[str]
    manual_title: Optional[str]
    source: str = "manual"  # "manual" | "auto"

pending_q: "queue.Queue[Job]" = queue.Queue()

# ========= ユーティリティ =========
def _ensure_package(pkg: str) -> None:
    if _is_frozen_bundle():
        raise RuntimeError(f"実行ファイル版には {pkg} が含まれていません。配布パッケージを再構築してください。")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def _has_ort() -> bool:
    try:
        import onnxruntime  # noqa
        return True
    except Exception:
        return False

def _level_from_chunk(chunk_i16: np.ndarray) -> float:
    if chunk_i16.size == 0: return 0.0
    rms = np.sqrt(np.mean((chunk_i16.astype(np.float32)/32768.0)**2) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-9)
    return max(0.0, min(100.0, (db + 60.0) / 60.0 * 100.0))

def _segment_len_sec() -> float:
    with state.lock:
        if not state.current_segment_frames or not state.samplerate or not state.channels:
            return 0.0
        total_samples = sum(len(a) for a in state.current_segment_frames)
    return total_samples / float(state.samplerate * state.channels)

def _trim_trailing_silence(frames_1d: np.ndarray, sr: int, ch: int,
                           thr_pct: float, pad_sec: float) -> np.ndarray:
    if frames_1d.size == 0: return frames_1d
    try:
        arr = frames_1d.reshape((-1, ch))
    except ValueError:
        arr = frames_1d.reshape((-1, 1)); ch = 1
    thr = int(32767 * (thr_pct / 100.0))
    active = np.any(np.abs(arr) > thr, axis=1)
    idx = np.where(active)[0]
    if idx.size == 0:
        return np.array([], dtype=np.int16)
    last = idx[-1]
    keep = min(arr.shape[0], last + 1 + int(pad_sec * sr))
    return arr[:keep].reshape(-1).astype(np.int16)

# ========= Whisper初期化（CUDA→CPU自動フォールバック）=========
_whisper_model: object | None = None
def _init_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        fw = importlib.import_module("faster_whisper")
    except ImportError:
        messagebox.showinfo("依存導入", "faster-whisper をインストールします。")
        _ensure_package("faster-whisper")
        fw = importlib.import_module("faster_whisper")
    WhisperModel = getattr(fw, "WhisperModel")

    def pick_compute(dev: str) -> str:
        if FW_COMPUTE_CFG:
            return FW_COMPUTE_CFG
        return "float16" if dev == "cuda" else "int8"

    tried = []
    pref = FW_DEVICE_CFG.lower()
    if pref == "auto":
        tried = [("cuda", pick_compute("cuda")), ("cpu", pick_compute("cpu"))]
    else:
        tried = [(pref, pick_compute(pref))]
        if pref != "cpu":
            tried.append(("cpu", pick_compute("cpu")))

    last_err = None
    for dev, comp in tried:
        try:
            _whisper_model = WhisperModel(FW_MODEL, device=dev, compute_type=comp)
            print(f"[INFO] faster-whisper init: model={FW_MODEL} device={dev} compute_type={comp}")
            return _whisper_model
        except Exception as e:
            print(f"[WARN] Whisper init failed on device={dev} compute={comp}: {e}")
            last_err = e
    raise RuntimeError(f"Whisperモデル初期化に失敗: {last_err}")

def get_whisper():
    return _init_whisper_model()

# ========= soundcard ロード =========
_sc_mod = None
def get_soundcard():
    global _sc_mod
    if _sc_mod is not None:
        return _sc_mod
    try:
        _sc_mod = importlib.import_module("soundcard")
    except ImportError:
        messagebox.showinfo("依存導入", "soundcard をインストールします。")
        _ensure_package("soundcard")
        _sc_mod = importlib.import_module("soundcard")
    return _sc_mod

# ========= デバイス列挙 =========
def _supports_loopback_flag() -> bool:
    try:
        if 'loopback' not in inspect.signature(sd.WasapiSettings).parameters:
            return False
        sd.WasapiSettings(loopback=True); return True
    except Exception:
        return False

def list_wasapi_output_devices_sd():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    wasapi_index = None
    for i, h in enumerate(hostapis):
        if (h.get("name") or "").lower().startswith("windows wasapi"):
            wasapi_index = i; break
    if wasapi_index is None: return []
    cand = []
    if _supports_loopback_flag():
        for idx, d in enumerate(devices):
            if d.get("hostapi")==wasapi_index and d.get("max_output_channels",0)>0:
                cand.append(("sd", idx, d))
    else:
        for idx, d in enumerate(devices):
            if d.get("hostapi")==wasapi_index and d.get("max_input_channels",0)>0:
                if "loopback" in (d.get("name") or "").lower():
                    cand.append(("sd", idx, d))
    return cand

def list_loopback_microphones_sc():
    sc = get_soundcard()
    names = [m.name for m in sc.all_microphones(include_loopback=True)]
    try:
        sp = sc.default_speaker()
        if sp and sp.name in names:
            names.sort(key=lambda n: 0 if n == sp.name else 1)
    except Exception:
        pass
    return names

# ========= 録音コールバック =========
def audio_callback(indata, frames, time_info, status):
    data = np.frombuffer(indata, dtype=np.int16).copy()
    with state.lock:
        state.current_segment_frames.append(data)
        level = _level_from_chunk(data)
        state.last_rms = level
        if state.samplerate:
            dur = float(frames) / float(state.samplerate)
            if level < AUTO_CUT_LEVEL_PCT:
                state.silence_run_sec += dur
            else:
                state.silence_run_sec = 0.0

def _sc_recording_loop(mic_name: str, samplerate: int, channels: int):
    ole32 = None
    try:
        if sys.platform == "win32":
            ole32 = ctypes.windll.ole32
            ole32.CoInitializeEx(None, 0x0)  # MTA
        sc = get_soundcard()
        mic = sc.get_microphone(mic_name, include_loopback=True)
        state.sc_stop.clear()
        with mic.recorder(samplerate=samplerate, channels=channels, blocksize=SC_BLOCKSIZE) as rec:
            while not state.sc_stop.is_set():
                data = rec.record(numframes=SC_BLOCKSIZE)
                chunk_i16 = np.clip(data, -1.0, 1.0)
                chunk_i16 = (chunk_i16 * 32767.0).astype(np.int16).reshape(-1)
                with state.lock:
                    state.current_segment_frames.append(chunk_i16)
                    level = _level_from_chunk(chunk_i16)
                    state.last_rms = level
                    dur = float(SC_BLOCKSIZE) / float(samplerate)
                    if level < AUTO_CUT_LEVEL_PCT:
                        state.silence_run_sec += dur
                    else:
                        state.silence_run_sec = 0.0
    finally:
        if ole32 is not None:
            try: ole32.CoUninitialize()
            except Exception: pass

# ========= 録音制御 =========
def start_recording(device_key: Tuple[str,str]):
    if state.running: return
    backend, ident = device_key
    state.backend = backend
    if backend == "sd":
        idx = int(ident.split(":")[1])
        info = sd.query_devices(idx)
        samplerate = int(info.get("default_samplerate") or 48000)
        if _supports_loopback_flag():
            channels = min(int(info.get("max_output_channels") or 2), 2)
            extra = sd.WasapiSettings(loopback=True)
        else:
            channels = min(int(info.get("max_input_channels") or 2), 2)
            extra = None
        state.device_index = idx; state.samplerate = samplerate; state.channels = channels
        state.reset_segment()
        stream = sd.RawInputStream(
            samplerate=samplerate, blocksize=BLOCKSIZE, device=idx,
            channels=channels, dtype="int16", callback=audio_callback, extra_settings=extra)
        stream.start(); state.stream = stream; state.running = True
        state.last_cut_at = time.time()
    else:
        mic_name = ident
        state.device_index = None; state.samplerate = 48000; state.channels = 2
        state.reset_segment()
        t = threading.Thread(target=_sc_recording_loop, args=(mic_name, state.samplerate, state.channels), daemon=True)
        state.sc_thread = t; t.start(); state.running = True
        state.last_cut_at = time.time()

def stop_recording():
    if not state.running: return
    if state.backend == "sd":
        if state.stream:
            try: state.stream.stop(); state.stream.close()
            except Exception: pass
        state.stream = None
    else:
        state.sc_stop.set()
        if state.sc_thread and state.sc_thread.is_alive():
            state.sc_thread.join(timeout=1.0)
        state.sc_thread = None
    state.running = False

def flush_segment_to_wav(auto: bool = False) -> Optional[str]:
    with state.lock:
        if not state.current_segment_frames:
            return None
        frames = np.concatenate(state.current_segment_frames)
        state.silence_run_sec = 0.0
        state.current_segment_frames = []

    sr = state.samplerate or 48000
    ch = state.channels or 1

    if auto:
        frames = _trim_trailing_silence(frames, sr, ch, thr_pct=AUTO_CUT_LEVEL_PCT, pad_sec=AUTO_TRIM_PAD_SEC)

    if frames.size == 0:
        return None

    total_sec = frames.size / float(sr * ch)
    if total_sec < AUTO_CUT_MIN_SEC:
        return None

    if not auto:
        level = _level_from_chunk(frames)
        if level < 5.0:
            raise RuntimeError("録音レベルが極端に低い/無音です。出力先の音量やミュート、デバイス選択を確認してください。")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    target_dir = AUDIO_OUTPUT_DIR
    os.makedirs(target_dir, exist_ok=True)
    wav_path = os.path.join(target_dir, f"segment-{ts}.wav")
    try:
        data = frames.reshape((-1, ch))
    except ValueError:
        data = frames.reshape((-1, 1))
    sf.write(wav_path, data, sr, subtype="PCM_16")
    return wav_path

# ========= 文字起こし（整形つき） =========
def _split_sentences_ja(s: str) -> List[str]:
    """日本語の簡易文分割（句点で行分割）。句点が無ければ1要素。"""
    s = re.sub(r"\s+", " ", s).strip()
    if not s: return []
    parts = re.split(r"(?<=[。．！？!?])", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p: continue
        out.append(p)
    return out

def transcribe_segments(file_path: str):
    """faster-whisper のセグメント（start/end/text）をリスト化して返す"""
    model = get_whisper()
    seg_iter, info = model.transcribe(
        file_path, language="ja", vad_filter=_has_ort(),
        vad_parameters={"min_silence_duration_ms": 300}, beam_size=5
    )
    segs = []
    for seg in seg_iter:
        # seg: .start .end .text
        text = (seg.text or "").strip()
        if text:
            segs.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return segs

def format_transcript_from_segments(segs: List[dict],
                                    para_gap_sec: float = PARA_GAP_SEC,
                                    sentence_fallback: int = SENTENCE_FALLBACK) -> str:
    """句点とセグメント間の無音ギャップから、読みやすい『文ごとの改行＋段落』を作る"""
    if not segs: return ""
    lines: List[str] = []
    buf = ""
    prev_end = None
    for s in segs:
        t = s["text"]
        st, ed = s["start"], s["end"]

        # 段落分け：前セグメントからのギャップが閾値以上
        if prev_end is not None and (st - prev_end) >= para_gap_sec:
            # まずバッファを文で吐き出す
            if buf.strip():
                for sent in _split_sentences_ja(buf):
                    lines.append(sent)
                buf = ""
            # 空行（段落区切り）
            if lines and lines[-1] != "":
                lines.append("")

        # バッファへ追加
        if buf:
            buf += " " + t
        else:
            buf = t

        # 句点が含まれていれば文で吐き出し
        if re.search(r"[。．！？!?]", buf):
            sents = _split_sentences_ja(buf)
            # 最後の文が句点で終わっていないときは、最後だけ次のセグメントへ持ち越し
            if sents and not re.search(r"[。．！？!?]$", sents[-1]):
                for sent in sents[:-1]:
                    lines.append(sent)
                buf = sents[-1]
            else:
                for sent in sents:
                    lines.append(sent)
                buf = ""
        else:
            # 句点が一つもない長文になってきたら、セグメント単位で改行（潰れ防止）
            if len(buf) >= sentence_fallback:
                lines.append(buf.strip())
                buf = ""

        prev_end = ed

    # 余りを吐き出し
    if buf.strip():
        lines.append(buf.strip())

    # 末尾の余計な空行を削る
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)

def transcribe_with_whisper_formatted(file_path: str) -> tuple[str, str]:
    """(raw_text, formatted_text) を返す"""
    segs = transcribe_segments(file_path)
    raw = "".join([s["text"] for s in segs]).strip()
    formatted = format_transcript_from_segments(segs)
    return raw, formatted

# ========= 要約（厳密JSON） =========
STRUCT_PROMPT = f"""
あなたは高校生向けノートを作る日本語要約関数です。
以下の制約を厳密に守り、**JSONオブジェクトのみ**を返してください。
先頭は `{{`、末尾は `}}`。説明文・前置き・コードブロック・マークダウンは禁止。

必須キー:
- "title": 12〜24字の短い日本語タイトル（固有名詞・重要語を含める。絵文字・記号×）
- "bullets": 3〜{max(3, min(7, SUM_BULLETS_MAX))} 個の配列。**各要素は1文の完全文の日本語文字列**で、文末は「。」で終える。
  * **禁止**: 名詞の羅列、ページ/行番号、授業運営の文言（例:「〜しましょう」「ポイントを確認」）、そのままの書き起こし。
  * 可能なら「誰が/何を/なぜ/どのように」を自然に含め、因果・数値・定義・結論を簡潔に入れる。
  * 明らかなASR誤変換は文脈で**軽く**補正（例: 栄養文脈の「支出」→「脂質」など）。
- "summary": **2〜4文の段落**（改行なし）。短い書き起こしの連結ではなく、情報を再構成して説明する。
  * 句読点を適切に補い、冗長・重複・羅列を避け、要点→理由/影響→補足の順で滑らかに。
- "subject": 次のいずれか1つ（厳密一致）
  {", ".join([s for s in SUBJECTS if s != "自動判定"])}

出力は**純粋なJSON**のみで、余計な文字は一切出力しないこと。

<TRANSCRIPT>
{{text}}
</TRANSCRIPT>
""".strip()

def _ollama_generate(model: str, prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",      # ★ JSON強制
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_predict": SUM_TOKENS,
                "repeat_penalty": 1.1
            }
        },
        timeout=180
    )
    if r.status_code == 404:
        raise RuntimeError("MODEL_NOT_FOUND")
    r.raise_for_status()
    return r.json().get("response", "").strip()

def _bullet_from_kv(d: dict) -> str:
    def v(*keys):
        for k in keys:
            if k in d and str(d[k]).strip():
                return str(d[k]).strip()
        return ""
    who   = v("who","だれ","誰","subject","対象")
    what  = v("what","何","内容","topic")
    when  = v("when","いつ","時間")
    where = v("where","どこ","場所")
    why   = v("why","理由","目的")
    how   = v("how","方法","手段","どのように")
    tp = "、".join([x for x in [when, where] if x])
    if who and what:
        core = f"{who}は" + (f"{tp}に{what}する" if tp else f"{what}する")
    elif what:
        core = (f"{tp}に{what}する" if tp else f"{what}する").lstrip("に")
    else:
        core = "、".join([x for x in [who, tp, why, how] if x]) or "重要事項"
    if why and how: tail = f"。目的は{why}、方法は{how}。"
    elif why:       tail = f"。目的は{why}。"
    elif how:       tail = f"。方法は{how}。"
    else:           tail = "。"
    return (core + tail).replace("、、", "、")

def _normalize_bullets(bullets_raw) -> list[str]:
    out = []
    if not isinstance(bullets_raw, list):
        bullets_raw = [bullets_raw]
    for b in bullets_raw:
        if isinstance(b, str):
            s = b.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    d = ast.literal_eval(s)
                    if isinstance(d, dict):
                        s = _bullet_from_kv(d)
                except Exception:
                    pass
            if s and not re.search(r"[。！？]$", s):
                s += "。"
            out.append(s)
        elif isinstance(b, dict):
            out.append(_bullet_from_kv(b))
        else:
            s = str(b).strip()
            if s and not re.search(r"[。！？]$", s):
                s += "。"
            out.append(s)
    # 空除去 & 重複軽減
    seen = set(); res = []
    for s in out:
        if s and s not in seen:
            res.append(s); seen.add(s)
    return res[:max(3, min(7, SUM_BULLETS_MAX))]

def _normalize_summary(s: str) -> str:
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s and not re.search(r"[。！？]$", s):
        s += "。"
    return s

def summarize_structured(text: str) -> dict:
    out = _ollama_generate(OLLAMA_MODEL, STRUCT_PROMPT.replace("{text}", text))
    try:
        data = json.loads(out)
    except Exception:
        return {"title":"概要", "bullets":[out[:120] + "。"], "summary":_normalize_summary(out[:400]), "subject":"その他"}
    title = str(data.get("title","概要")).strip()
    bullets = _normalize_bullets(data.get("bullets", []))
    summary = _normalize_summary(str(data.get("summary","")).strip())
    subject = str(data.get("subject","その他")).strip()
    if subject not in SUBJECTS: subject = "その他"
    return {"title":title, "bullets":bullets, "summary":summary, "subject":subject}

# ========= 保存ユーティリティ =========
def sanitize_filename(name: str, maxlen: int = 60) -> str:
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r'[\\/:*?"<>|]+', " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:maxlen] if len(name)>maxlen else name

def ensure_unique(path: str) -> str:
    if not os.path.exists(path): return path
    root, ext = os.path.splitext(path)
    k = 2
    while True:
        p = f"{root}_{k}{ext}"
        if not os.path.exists(p): return p
        k += 1

def write_segment_markdown(subject: str, ts: str, title: str,
                           transcript_formatted: str, bullets: List[str],
                           summary: str, wav_path: str):
    summary_root = SUMMARY_OUTPUT_DIR
    os.makedirs(summary_root, exist_ok=True)
    subj_dir = os.path.join(summary_root, subject)
    os.makedirs(subj_dir, exist_ok=True)
    safe_title = sanitize_filename(title or "タイトルなし")
    md_name = f"{subject}_{ts}_{safe_title}.md"
    md_path = ensure_unique(os.path.join(subj_dir, md_name))
    try:
        rel_wav = os.path.relpath(wav_path, start=subj_dir)
    except ValueError:
        rel_wav = wav_path

    header = [
        f"# {subject} / {title}",
        f"- 収録: {ts}",
        f"- 音声: [{os.path.basename(wav_path)}]({rel_wav})",
        f"- Whisper: {FW_MODEL}",
        f"- 要約モデル: {OLLAMA_MODEL}",
        "",
        "## 重要点",
        *[f"- {b}" for b in bullets],
        "",
        "## 要約",
        summary,
        "",
        "## 文字起こし",
        transcript_formatted
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header))

    os.makedirs(os.path.dirname(INDEX_MD) or SUMMARY_OUTPUT_DIR, exist_ok=True)
    with open(INDEX_MD, "a", encoding="utf-8") as f:
        try:
            rel_md = os.path.relpath(md_path, start=SUMMARY_OUTPUT_DIR)
        except ValueError:
            rel_md = md_path
        f.write(f"- {ts} [{subject}] {title} -> {rel_md}\n")

    return md_path

# ========= バックグラウンド処理 =========
def transcribe_with_whisper_file_then_save(job, log_widget: tk.Text):
    try:
        src = "自動" if job.source == "auto" else "手動"
        log_widget_insert(log_widget, f"[INFO] ({src}) キュー処理: {os.path.basename(job.wav_path)}\n")
        raw_text, formatted_text = transcribe_with_whisper_formatted(job.wav_path)
        log_widget_insert(log_widget, f"[OK] 文字起こし（{len(raw_text)}文字 / 整形後 {len(formatted_text)}文字）\n")
        log_widget_insert(log_widget, "[INFO] 要約+科目判定+タイトル 生成中...\n")
        struct = summarize_structured(raw_text)  # ← 要約は原文（改行なし）で
        if job.manual_title:
            struct["title"] = job.manual_title.strip()
        subject = job.chosen_subject or struct.get("subject") or "その他"
        if subject not in SUBJECTS: subject = "その他"
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        md_path = write_segment_markdown(
            subject=subject, ts=ts, title=struct["title"],
            transcript_formatted=formatted_text,
            bullets=struct.get("bullets", []),
            summary=struct.get("summary",""), wav_path=job.wav_path
        )
        log_widget_insert(log_widget, f"[DONE] 出力: {md_path}\n")
    except Exception as e:
        log_widget_insert(log_widget, f"[ERROR] {e}\n")

def worker_loop(log_widget: tk.Text):
    while True:
        job: Job = pending_q.get()
        try:
            transcribe_with_whisper_file_then_save(job, log_widget)
        finally:
            pending_q.task_done()

# ========= GUI =========
def log_widget_insert(widget: tk.Text, msg: str):
    widget.configure(state="normal"); widget.insert("end", msg); widget.see("end"); widget.configure(state="disabled")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chrome Whisper Transcriber (Readable Transcript)")
        self.geometry("1000x760"); self.resizable(True, True)

        # デバイス列挙
        sd_devs = list_wasapi_output_devices_sd()
        self.dev_entries: List[Tuple[str,str,str]] = []
        self.last_device_key: Optional[Tuple[str, str]] = None
        if sd_devs:
            for _, idx, d in sd_devs:
                self.dev_entries.append(("sd", f"idx:{idx}", f"[SD] {d['name']} (idx:{idx})"))
        else:
            try: sc_names = list_loopback_microphones_sc()
            except Exception: sc_names = []
            for n in sc_names:
                self.dev_entries.append(("sc", n, f"[SC] {n} (loopback)"))
        if not self.dev_entries:
            messagebox.showerror("エラー","ループバック録音デバイスが見つかりません。"); self.destroy(); return

        # 上段：デバイス
        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        ttk.Label(top, text="録音対象デバイス:").pack(side="left")
        values = [lbl for _,_,lbl in self.dev_entries]
        self.combo = ttk.Combobox(top, state="readonly", width=64, values=values)
        self.combo.current(0); self.combo.pack(side="left", padx=8)

        # 科目＆タイトル
        meta = ttk.Frame(self, padding=8); meta.pack(fill="x")
        ttk.Label(meta, text="科目:").pack(side="left")
        self.subject_var = tk.StringVar(value=SUBJECTS[0])
        self.subject_combo = ttk.Combobox(meta, state="readonly", width=24, values=SUBJECTS, textvariable=self.subject_var)
        self.subject_combo.pack(side="left", padx=(4,16))
        ttk.Label(meta, text="タイトル（任意）:").pack(side="left")
        self.title_var = tk.StringVar(value="")
        self.title_entry = ttk.Entry(meta, textvariable=self.title_var, width=40)
        self.title_entry.pack(side="left", padx=4)

        saving = ttk.LabelFrame(self, text="Output folders", padding=8)
        saving.pack(fill="x", padx=8, pady=(0, 8))
        saving.columnconfigure(1, weight=1)

        ttk.Label(saving, text="Audio files:").grid(row=0, column=0, sticky="w")
        self.audio_dir_var = tk.StringVar(value=AUDIO_OUTPUT_DIR)
        audio_entry = ttk.Entry(saving, textvariable=self.audio_dir_var, state="readonly", width=80)
        audio_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(saving, text="Browse...", command=self.on_choose_audio_dir).grid(row=0, column=2, padx=4)

        ttk.Label(saving, text="Markdown/summary:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.summary_dir_var = tk.StringVar(value=SUMMARY_OUTPUT_DIR)
        summary_entry = ttk.Entry(saving, textvariable=self.summary_dir_var, state="readonly", width=80)
        summary_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Button(saving, text="Browse...", command=self.on_choose_summary_dir).grid(row=1, column=2, padx=4, pady=(4, 0))

        # 自動区切りの状態表示
        auto = ttk.Frame(self, padding=(8,0)); auto.pack(fill="x")
        self.auto_enabled = tk.BooleanVar(value=AUTO_CUT_ENABLED)
        ttk.Checkbutton(auto, text=f"無音 {AUTO_CUT_SILENCE_SEC:.1f} 秒で自動区切り", variable=self.auto_enabled).pack(side="left")

        # ボタン群
        btn = ttk.Frame(self, padding=8); btn.pack(fill="x")
        self.btn_start = ttk.Button(btn, text="録音開始", command=self.on_start)
        self.btn_cut   = ttk.Button(btn, text="区切って書き起こし/保存", command=self.on_cut, state="disabled")
        self.btn_stop  = ttk.Button(btn, text="停止", command=self.on_stop, state="disabled")
        self.btn_open_audio = ttk.Button(btn, text="音声フォルダを開く", command=self.on_open_audio)
        self.btn_open  = ttk.Button(btn, text="要約フォルダを開く", command=self.on_open)
        for b in (self.btn_start, self.btn_cut, self.btn_stop, self.btn_open_audio, self.btn_open):
            b.pack(side="left", padx=4)
        self.btn_resume = ttk.Button(btn, text="録音再開", command=self.on_resume, state="disabled")

        # レベルメーター
        meter = ttk.Frame(self, padding=(8,0)); meter.pack(fill="x")
        ttk.Label(meter, text="録音レベル:").pack(side="left")
        self.pb = ttk.Progressbar(meter, orient="horizontal", length=360, mode="determinate", maximum=100); self.pb.pack(side="left", padx=8)
        self.lbl_db = ttk.Label(meter, text="-- %"); self.lbl_db.pack(side="left")

        # ログ
        frame_log = ttk.LabelFrame(self, text="ログ", padding=8)
        frame_log.pack(fill="both", expand=True, padx=8, pady=8)
        self.log = tk.Text(frame_log, height=18, state="disabled"); self.log.pack(fill="both", expand=True)

        note = (
            "Note: transcripts are saved as Markdown with sentence-based breaks.\n"
            "Note: use the Output folders panel to change audio and Markdown destinations."
        )
        ttk.Label(self, text=note, foreground="#444").pack(padx=8, pady=(0,6))

        # ワーカー起動
        self.worker = threading.Thread(target=worker_loop, args=(self.log,), daemon=True); self.worker.start()
        self.after(150, self._update_meter)

        # 自動区切り監視スレッド
        self.auto_stop = threading.Event()
        self.auto_thread: Optional[threading.Thread] = None

    def _update_meter(self):
        with state.lock: val = state.last_rms
        self.pb["value"] = val; self.lbl_db.configure(text=f"{val:5.1f} %")
        self.after(150, self._update_meter)

    def _selected_device_key(self) -> Tuple[str,str]:
        i = self.combo.current()
        backend, key, _ = self.dev_entries[i]
        return backend, key

    def set_running_ui(self, running: bool):
        self.btn_start.configure(state="disabled" if running else "normal")
        self.btn_cut.configure(state="normal" if running else "disabled")
        if running:
            self._show_stop_button()
            self.btn_stop.configure(state="normal")
            self.btn_resume.configure(state="disabled")
        else:
            self.btn_stop.configure(state="disabled")
            self.btn_resume.configure(state="disabled")

    def on_choose_audio_dir(self):
        initial = AUDIO_OUTPUT_DIR if os.path.isdir(AUDIO_OUTPUT_DIR) else SUMMARY_OUTPUT_DIR
        selected = filedialog.askdirectory(initialdir=initial, title="Select audio output folder")
        if not selected:
            return
        try:
            new_dir = set_audio_output_dir(selected)
        except Exception as e:
            messagebox.showerror("Failed to set audio output folder", str(e))
            return
        self.audio_dir_var.set(new_dir)
        if hasattr(self, "log"):
            log_widget_insert(self.log, f"[CFG] audio_dir -> {new_dir}\n")

    def on_choose_summary_dir(self):
        initial = SUMMARY_OUTPUT_DIR if os.path.isdir(SUMMARY_OUTPUT_DIR) else AUDIO_OUTPUT_DIR
        selected = filedialog.askdirectory(initialdir=initial, title="Select Markdown/summary folder")
        if not selected:
            return
        try:
            new_dir = set_summary_output_dir(selected)
        except Exception as e:
            messagebox.showerror("Failed to set Markdown/summary folder", str(e))
            return
        self.summary_dir_var.set(new_dir)
        if hasattr(self, "log"):
            log_widget_insert(self.log, f"[CFG] summary_dir -> {new_dir}\n")

    def _open_directory(self, path: str):
        target = os.path.abspath(path)
        os.makedirs(target, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(target)
        else:
            filedialog.askopenfilename(initialdir=target)

    def on_open_audio(self):
        self._open_directory(AUDIO_OUTPUT_DIR)

    def _show_stop_button(self):
        if self.btn_resume.winfo_manager():
            self.btn_resume.pack_forget()
        if not self.btn_stop.winfo_manager():
            self.btn_stop.pack(side="left", padx=4, before=self.btn_open)

    def _show_resume_button(self):
        if self.btn_stop.winfo_manager():
            self.btn_stop.pack_forget()
        if not self.btn_resume.winfo_manager():
            self.btn_resume.pack(side="left", padx=4, before=self.btn_open)

    def on_start(self):
        try:
            get_whisper()
        except Exception as e:
            messagebox.showerror("Whisper初期化失敗", str(e)); return
        try:
            backend, key = self._selected_device_key()
            start_recording((backend, key))
            self.last_device_key = (backend, key)
            self.set_running_ui(True)
            label = self.dev_entries[self.combo.current()][2]
            log_widget_insert(self.log, f"[START] 録音開始（{label}） SR={state.samplerate} CH={state.channels} backend={state.backend}\n")
            # 自動区切り監視開始
            self.auto_stop.clear()
            self.auto_thread = threading.Thread(target=self._auto_cut_loop, daemon=True)
            self.auto_thread.start()
        except Exception as e:
            messagebox.showerror("録音失敗", str(e))

    def on_stop(self):
        stop_recording()
        self.set_running_ui(False)
        self.auto_stop.set()
        if self.auto_thread and self.auto_thread.is_alive():
            self.auto_thread.join(timeout=1.0)
        log_widget_insert(self.log, "[STOP] 録音停止（キュー中の処理は継続）\n")
        self._show_resume_button()
        resume_state = "normal" if self.last_device_key else "disabled"
        self.btn_resume.configure(state=resume_state)

    def on_resume(self):
        if not self.last_device_key:
            messagebox.showwarning("\u9332\u97f3\u518d\u958b\u306b\u5931\u6557", "\u518d\u958b\u3067\u304d\u308b\u9332\u97f3\u304c\u3042\u308a\u307e\u305b\u3093\u3002")
            return
        backend, key = self.last_device_key
        label = None
        for idx, entry in enumerate(self.dev_entries):
            if entry[0] == backend and entry[1] == key:
                self.combo.current(idx)
                label = entry[2]
                break
        if label is None:
            label = f"{backend}:{key}"
        try:
            start_recording((backend, key))
            self.set_running_ui(True)
            self.auto_stop.clear()
            self.auto_thread = threading.Thread(target=self._auto_cut_loop, daemon=True)
            self.auto_thread.start()
            log_widget_insert(self.log, f"[RESUME] \u9332\u97f3\u518d\u958b ({label}) SR={state.samplerate} CH={state.channels} backend={state.backend}\n")
        except Exception as e:
            messagebox.showerror("\u9332\u97f3\u518d\u958b\u306b\u5931\u6557", str(e))
            self.set_running_ui(False)
            self._show_resume_button()
            self.btn_resume.configure(state="normal")

    def _queue_job(self, wav_path: str, source: str):
        subj = self.subject_var.get()
        chosen_subject = None if subj == "自動判定" else subj
        manual_title = (self.title_var.get() or "").strip() or None
        pending_q.put(Job(wav_path=wav_path, chosen_subject=chosen_subject, manual_title=manual_title, source=source))

    def on_cut(self):
        try:
            wav_path = flush_segment_to_wav(auto=False)
            if not wav_path:
                log_widget_insert(self.log, "[WARN] セグメントが短すぎます/空です\n"); return
            self._queue_job(wav_path, source="manual")
            log_widget_insert(self.log, f"[SEGMENT] 手動保存: {os.path.basename(wav_path)} / 科目={self.subject_var.get()} / タイトル={self.title_var.get() or '（自動）'}\n")
        except Exception as e:
            messagebox.showwarning("セグメント保存問題", str(e)); log_widget_insert(self.log, f"[WARN] {e}\n")

    def _auto_cut_loop(self):
        # 200msごとに無音継続を監視
        while not self.auto_stop.is_set():
            time.sleep(0.2)
            if not self.auto_enabled.get() or not state.running:
                continue
            with state.lock:
                silence = state.silence_run_sec
            if silence >= AUTO_CUT_SILENCE_SEC:
                if _segment_len_sec() >= AUTO_CUT_MIN_SEC:
                    wav_path = flush_segment_to_wav(auto=True)
                    if wav_path:
                        self._queue_job(wav_path, source="auto")
                        log_widget_insert(self.log, f"[AUTO-CUT] 無音 {silence:.1f}s -> 保存: {os.path.basename(wav_path)}（科目={self.subject_var.get()}）\n")

    def on_open(self):
        self._open_directory(SUMMARY_OUTPUT_DIR)

def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
        except Exception: pass
    app = App(); app.mainloop()

if __name__ == "__main__":
    main()
