#!/usr/bin/env python3
"""Utility to download Ollama models (default: llama3:8b)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence
from urllib.parse import urljoin, urlparse

DEFAULT_SERVICE_URL = "http://127.0.0.1:11434"
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def _candidate_ollama_paths() -> list[Path]:
    """Return likely ollama.exe locations on Windows."""
    candidates: list[Path] = []
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        candidates.append(Path(local_app) / "Programs" / "Ollama" / "ollama.exe")
    program_files = os.environ.get("ProgramW6432") or "C:/Program Files"
    candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        candidates.append(Path(program_files_x86) / "Ollama" / "ollama.exe")
    home = Path.home()
    candidates.append(home / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe")
    return candidates


def discover_ollama_executable(hint: str) -> str | None:
    """Resolve the Ollama executable path from PATH or common install locations."""
    resolved = shutil.which(hint)
    if resolved:
        return resolved

    expanded = Path(os.path.expanduser(os.path.expandvars(hint)))
    if expanded.exists():
        return str(expanded)

    if os.name == "nt":
        for candidate in _candidate_ollama_paths():
            if candidate.exists():
                return str(candidate)

    return None


def normalize_service_url(raw: str | None) -> str:
    """Normalize the Ollama service URL, applying defaults and adding scheme."""
    if not raw:
        return DEFAULT_SERVICE_URL
    raw = raw.strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return f"http://{raw}"


def service_env_override(url: str, env: dict[str, str]) -> None:
    """Populate OLLAMA_HOST for child processes based on the chosen URL."""
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        host_value = parsed.netloc
    else:
        host_value = url.lstrip("http://").lstrip("https://")
    env.setdefault("OLLAMA_HOST", host_value)


def service_is_ready(url: str, timeout: float = 1.5) -> bool:
    """Return True if the Ollama HTTP API responds within the timeout."""
    probe = urljoin(url, "/api/tags")
    request = urllib.request.Request(probe, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def ensure_service(ollama_path: str, url: str, autostart: bool, startup_timeout: float) -> bool:
    """Ensure the Ollama service is reachable, optionally autostarting it."""
    if service_is_ready(url):
        return True

    if not autostart:
        print(
            "[WARN] Ollama service is unreachable and auto-start is disabled. The pull command may fail.",
            file=sys.stderr,
        )
        return False

    print("[INFO] Ollama service not detected. Starting `ollama serve` in the background...")

    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    try:
        subprocess.Popen([ollama_path, "serve"], **kwargs)
    except OSError as exc:
        print(f"[ERROR] Failed to launch `ollama serve`: {exc}", file=sys.stderr)
        return False

    deadline = time.time() + max(1.0, startup_timeout)
    while time.time() < deadline:
        if service_is_ready(url):
            print("[INFO] Ollama service is now reachable.")
            return True
        time.sleep(0.5)

    print(
        "[ERROR] Timed out waiting for the Ollama service to become ready.",
        file=sys.stderr,
    )
    return False


def run(cmd: Sequence[str], *, env: dict[str, str] | None = None) -> int:
    """Run a command while streaming stdout/stderr to the user."""
    return subprocess.call(cmd, env=env)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trigger an Ollama model download via `ollama pull`."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="llama3:8b",
        help="Model identifier to pull (default: %(default)s)",
    )
    parser.add_argument(
        "--ollama",
        default="ollama",
        help="Path to the Ollama executable (auto-detect by default)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Ollama host/URL (default: env OLLAMA_URL/OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Do not attempt to launch `ollama serve` automatically.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the Ollama service when auto-started (default: %(default)s)",
    )

    args = parser.parse_args(argv)

    ollama_path = discover_ollama_executable(args.ollama)
    if ollama_path is None:
        print(
            "[ERROR] Could not find the Ollama CLI. Install it or pass --ollama with the full path.",
            file=sys.stderr,
        )
        if os.name == "nt":
            print(
                "        Checked PATH and common locations such as %LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe",
                file=sys.stderr,
            )
        return 1

    service_url = normalize_service_url(
        args.host or os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST")
    )

    if ensure_service(ollama_path, service_url, not args.no_autostart, args.startup_timeout) is False:
        print("[ERROR] Aborting because the Ollama service is unavailable.", file=sys.stderr)
        return 1

    env = os.environ.copy()
    service_env_override(service_url, env)

    cmd = [ollama_path, "pull", args.model]
    print(f"[INFO] Running: {' '.join(cmd)}")
    print(f"[INFO] Target service: {service_url}")

    try:
        return_code = run(cmd, env=env)
    except KeyboardInterrupt:
        print("[WARN] Pull interrupted by user.", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"[ERROR] Failed to execute `{ollama_path}`: {exc}", file=sys.stderr)
        return 1

    if return_code != 0:
        print(
            f"[ERROR] Ollama exited with status {return_code}. Check the Ollama service logs for details.",
            file=sys.stderr,
        )
    else:
        print(f"[SUCCESS] Model `{args.model}` is ready.")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
