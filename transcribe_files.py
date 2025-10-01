# -*- coding: utf-8 -*-
# transcribe_files.py
import os
import sys
import glob
import datetime as dt
import importlib
import subprocess
import tempfile
from typing import List

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

# === 出力先（GUIと同じ） ===
BASE_DIR = _app_base_dir()
OUTPUT_DIR = _resolve_output_dir()
MASTER_MD = os.path.join(OUTPUT_DIR, "transcript_and_summary.md")

# === モデル設定（GUIと同じ。環境変数で上書き可） ===
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b-instruct")

FW_MODEL   = os.environ.get("FW_MODEL",   "medium")
FW_DEVICE  = os.environ.get("FW_DEVICE",  "cpu")        # WindowsはCPU運用が堅実
FW_COMPUTE = os.environ.get("FW_COMPUTE", "int8")       # 速度重視: int8 / 少し精度: int8_float16

# === 依存の遅延ロード ===
_fw_model = None

def _ensure(pkg: str):
    if getattr(sys, "frozen", False):
        raise RuntimeError(f"実行ファイル版には {pkg} が含まれていません。配布パッケージを再構築してください。")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def _has_ort() -> bool:
    try:
        import onnxruntime  # noqa
        return True
    except Exception:
        return False

def get_whisper():
    global _fw_model
    if _fw_model is not None:
        return _fw_model
    try:
        fw = importlib.import_module("faster_whisper")
    except ImportError:
        print("[INFO] faster-whisper が見つかりません。インストールします...")
        _ensure("faster-whisper")
        fw = importlib.import_module("faster_whisper")
    WhisperModel = getattr(fw, "WhisperModel")
    _fw_model = WhisperModel(FW_MODEL, device=FW_DEVICE, compute_type=FW_COMPUTE)
    return _fw_model

# === 文字起こし ===
def transcribe(file_path: str) -> str:
    model = get_whisper()
    use_vad = _has_ort()  # onnxruntime がある時だけ VAD を使う
    segments, info = model.transcribe(
        file_path,
        language="ja",
        vad_filter=use_vad,
        vad_parameters={"min_silence_duration_ms": 300},
        beam_size=5,
    )
    return "".join(seg.text for seg in segments).strip()

# === 要約（Ollama） ===
def summarize(text: str) -> str:
    prompt = (
        "You are a Japanese summarization assistant.\n"
        "Read the following transcript and respond in Japanese while following the requirements and output format.\n"
        "Requirements:\n"
        "1. List at least five key points in natural Japanese sentences that reflect the transcript, preserving proper nouns and numbers.\n"
        "2. Rephrase the content for each key point instead of copying a sentence fragment verbatim from the transcript.\n"
        "3. For the summary, do not use arrays, brackets, bullet points, Markdown lists, or quoted list representations (e.g., ['sentence1', 'sentence2']).\n"
        "4. Write the summary as one or two paragraphs consisting of 3 to 5 full Japanese sentences ending with the character \u3002, and provide enough context and background.\n"
        "5. Keep the overlap between the key points and the summary minimal.\n"
        "Output format:\n"
        "## \u91cd\u8981\u70b9\n"
        "- (Key point 1 in Japanese)\n"
        "- (Key point 2 in Japanese)\n"
        "- (Key point 3 in Japanese)\n"
        "- (Key point 4 in Japanese)\n"
        "- (Key point 5 in Japanese)\n"
        "## \u8981\u7d04\n"
        "(Summary in natural Japanese sentences written as plain text paragraphs without any list syntax)\n"
        "Replace the placeholder lines above with your results.\n"
        "----\n" + text + "\n----"
    )
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_predict": 512,
                "repeat_penalty": 1.1,
            },
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

# === Markdown 追記 ===
def append_markdown(src_path: str, transcript: str, summary: str):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = os.path.basename(src_path)
    block = (
        f"\n\n---\n### 既存ファイル {name}（{ts}）\n"
        f"**文字起こし**\n\n{transcript}\n\n"
        f"**要約**\n\n{summary}"
    )
    if not os.path.exists(MASTER_MD):
        with open(MASTER_MD, "w", encoding="utf-8") as f:
            f.write("# 文字起こし & 要約（既存音声のバッチ処理）\n")
            f.write(f"- 生成開始: {ts}\n")
            f.write(block)
    else:
        with open(MASTER_MD, "a", encoding="utf-8") as f:
            f.write(block)

# === 入力リストを作る ===
AUDIO_EXTS = {".wav",".mp3",".m4a",".mp4",".aac",".flac",".ogg",".opus",".webm",".mkv"}

def collect_inputs(args: List[str]) -> List[str]:
    paths: List[str] = []
    for a in args:
        p = os.path.expanduser(a)
        if os.path.isdir(p):
            # フォルダ配下の対象拡張子を再帰で拾う
            for ext in AUDIO_EXTS:
                paths += glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True)
        elif os.path.isfile(p):
            if os.path.splitext(p)[1].lower() in AUDIO_EXTS:
                paths.append(p)
    # 重複除去＆ソート
    dedup = sorted(set(os.path.abspath(x) for x in paths))
    return dedup

def main():
    if len(sys.argv) < 2:
        print("使い方: python transcribe_files.py <ファイル or フォルダ> [他にも可]")
        sys.exit(1)

    inputs = collect_inputs(sys.argv[1:])
    if not inputs:
        print("対象の音声ファイルが見つかりませんでした。対応拡張子:", ", ".join(sorted(AUDIO_EXTS)))
        sys.exit(2)

    print(f"[INFO] {len(inputs)} 件を処理します。")
    ok, ng = 0, 0
    for i, path in enumerate(inputs, 1):
        try:
            print(f"[{i}/{len(inputs)}] 文字起こし: {path}")
            text = transcribe(path)
            print(f"  -> OK 文字数: {len(text)}  要約中…")
            summary = summarize(text)
            append_markdown(path, text, summary)
            print("  -> DONE 追記:", MASTER_MD)
            ok += 1
        except Exception as e:
            print("  -> ERROR", e)
            ng += 1

    print(f"[RESULT] OK={ok} / NG={ng}  出力: {MASTER_MD}")

if __name__ == "__main__":
    main()
