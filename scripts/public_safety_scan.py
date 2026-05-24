"""Fail-fast public release scan for tracked repository files.

This script is intentionally conservative. It scans only tracked files so local
ignored runtime data does not cause noise, while still protecting the public
repository surface.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF_RELATIVE_PATH = "scripts/public_safety_scan.py"

BLOCKED_PATH_PATTERNS = [
    re.compile(r"(^|/)audio/"),
    re.compile(r"(^|/)speaker_embedding/"),
    re.compile(r"(^|/)data/(?!\.gitkeep$)"),
    re.compile(r"(^|/)logs?/"),
    re.compile(r"\.(wav|mp3|flac|m4a|npy|npz|onnx|pt|pth|ckpt)$", re.I),
]

BLOCKED_TEXT_PATTERNS = [
    ("github_token", re.compile(r"(gho|ghp|github_pat)_[A-Za-z0-9_]+")),
    ("private_key", re.compile(r"BEGIN (RSA|DSA|EC|OPENSSH|PRIVATE) KEY")),
    ("internal_ip", re.compile(r"\b10\.30\.\d{1,3}\.\d{1,3}\b")),
    ("old_workspace_path", re.compile(r"/workspace/project/audio_emr")),
    ("old_cuda_pin", re.compile(r"\bcuda:3\b")),
    ("old_demo_speaker_gdg", re.compile(r"\bgdg\b", re.I)),
    ("old_demo_speaker_yq", re.compile(r"\byq\b", re.I)),
    ("old_test_doctor", re.compile(r"\btest_doctor\b", re.I)),
    ("cn_id_label", re.compile(r"(身份证|住院号|病历号|医保号|患者姓名)")),
    ("cn_mobile_number", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
]

TEXT_EXTENSIONS = {
    ".cfg",
    ".css",
    ".env",
    ".example",
    ".gitignore",
    ".gitattributes",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def git_ls_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name in {
        "LICENSE",
        "VERSION",
        "SECURITY.md",
    }


def main() -> int:
    failures: list[str] = []
    tracked_files = git_ls_files()

    for relative in tracked_files:
        normalized = relative.replace("\\", "/")
        for pattern in BLOCKED_PATH_PATTERNS:
            if pattern.search(normalized):
                failures.append(f"blocked path: {relative}")

        path = ROOT / relative
        if normalized == SELF_RELATIVE_PATH:
            continue
        if not is_text_file(path):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-utf8 tracked text-like file: {relative}")
            continue

        for name, pattern in BLOCKED_TEXT_PATTERNS:
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                failures.append(f"{name}: {relative}:{line_number}")

    if failures:
        print("Public safety scan failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Public safety scan passed for {len(tracked_files)} tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
