import sys
import types
import os
import queue
import tempfile
import unittest


def _ensure_stub(name, attrs):
    if name in sys.modules:
        module = sys.modules[name]
    else:
        module = types.ModuleType(name)
        sys.modules[name] = module
    for key, value in attrs.items():
        setattr(module, key, value)


_ensure_stub("numpy", {
    "ndarray": type("ndarray", (), {}),
})
_ensure_stub("sounddevice", {
    "RawInputStream": type("RawInputStream", (), {}),
    "WasapiSettings": type("WasapiSettings", (), {}),
    "query_devices": lambda *args, **kwargs: [],
    "query_hostapis": lambda *args, **kwargs: [],
})
_ensure_stub("soundfile", {
    "write": lambda *args, **kwargs: None,
})
_ensure_stub("requests", {
    "post": lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError()),
})

from chrome_whisper_transcriber_local import (
    App,
    SUMMARY_OUTPUT_DIR,
    set_summary_output_dir,
    write_segment_markdown,
    pending_q,
)


class _Var:
    def __init__(self, value: str):
        self._value = value

    def get(self) -> str:
        return self._value


class WriteSegmentMarkdownTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_summary = SUMMARY_OUTPUT_DIR
        self._tmp = tempfile.TemporaryDirectory()
        set_summary_output_dir(self._tmp.name)

    def tearDown(self) -> None:
        set_summary_output_dir(self._original_summary)
        self._tmp.cleanup()

    def _create_wav(self, name: str) -> str:
        path = os.path.join(self._tmp.name, name)
        with open(path, "wb") as fh:
            fh.write(b"RIFFTEST")
        return path

    def test_audio_link_present_when_kept(self) -> None:
        wav_path = self._create_wav("kept.wav")
        md_path = write_segment_markdown(
            subject="数学",
            ts="20240101-000000",
            title="テスト",
            transcript_formatted="本文",
            bullets=["ポイント"],
            summary="要約",
            wav_path=wav_path,
            audio_kept=True,
        )
        with open(md_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("音声: [", content)
        self.assertIn("kept.wav", content)

    def test_audio_note_when_deleted(self) -> None:
        wav_path = self._create_wav("removed.wav")
        md_path = write_segment_markdown(
            subject="数学",
            ts="20240101-000001",
            title="テスト",
            transcript_formatted="本文",
            bullets=["ポイント"],
            summary="要約",
            wav_path=wav_path,
            audio_kept=False,
        )
        with open(md_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("音声: （処理後に削除）", content)
        self.assertNotIn("[removed.wav]", content)


class QueueJobDeleteToggleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._drain_queue()

    def tearDown(self) -> None:
        self._drain_queue()

    def _drain_queue(self) -> None:
        while True:
            try:
                pending_q.get_nowait()
            except queue.Empty:
                break

    def _make_app(self, keep_audio: bool, title: str = "") -> App:
        app = App.__new__(App)
        app.subject_var = _Var("数学")
        app.title_var = _Var(title)
        app.keep_audio_var = _Var(keep_audio)
        return app

    def test_queue_job_respects_keep_audio_setting(self) -> None:
        app = self._make_app(keep_audio=True)
        app._queue_job("dummy.wav", source="manual")
        job = pending_q.get_nowait()
        self.assertFalse(job.delete_after)

    def test_queue_job_marks_for_deletion_when_toggle_off(self) -> None:
        app = self._make_app(keep_audio=False)
        app._queue_job("dummy.wav", source="manual")
        job = pending_q.get_nowait()
        self.assertTrue(job.delete_after)

    def test_queue_job_override_prevents_deletion(self) -> None:
        app = self._make_app(keep_audio=False)
        app._queue_job("dummy.wav", source="import", delete_after=False)
        job = pending_q.get_nowait()
        self.assertFalse(job.delete_after)


if __name__ == "__main__":
    unittest.main()
