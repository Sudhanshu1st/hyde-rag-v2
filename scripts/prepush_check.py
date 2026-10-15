"""Pre-push safety check (installed as .git/hooks/pre-push). Blocks a push if:

* a secret-looking string or a .env file is tracked;
* restricted or downloaded data is tracked (restricted PDFs, BEIR data, SciFact labels, logs);
* a tracked file is larger than 50 MB (GitHub's warning threshold);
* any commit being pushed has an author/committer email other than the project's no-reply one;
* a locally blocklisted term appears in tracked text (terms live in .git/info/prepush-blocklist,
  one per line, so the list itself is never committed).

Usage: python scripts/prepush_check.py   (exit code 1 = blocked)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALLOWED_EMAIL = "109865453+Sudhanshu1st@users.noreply.github.com"
FORBIDDEN_PATHS = re.compile(
    r"^(\.env$|\.env\.(?!example$)|data/food/raw_restricted/|data/food/processed/|data/beir/|data/indexes/"
    r"|data/scifact_labels/|logs/)|\.log$"
)
SECRETS = re.compile(
    r"gsk_[A-Za-z0-9]{20,}|\bsk-(?:proj-|ant-)?[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
MAX_BYTES = 50 * 1024 * 1024


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def main() -> int:
    problems: list[str] = []
    files = [f for f in git("ls-files").splitlines() if f]

    for f in files:
        if FORBIDDEN_PATHS.search(f):
            problems.append(f"forbidden path tracked: {f}")
        path = REPO / f
        if path.exists() and path.stat().st_size > MAX_BYTES:
            problems.append(f"file over 50 MB: {f}")

    blocklist_file = REPO / ".git" / "info" / "prepush-blocklist"
    blocked = []
    if blocklist_file.exists():
        blocked = [
            t.strip().lower() for t in blocklist_file.read_text(encoding="utf-8").splitlines() if t.strip()
        ]

    for f in files:
        path = REPO / f
        if f == "uv.lock" or not path.exists() or path.suffix.lower() in {".pdf", ".gz", ".npy", ".npz"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if m := SECRETS.search(text):
            problems.append(f"possible secret in {f}: {m.group(0)[:12]}...")
        low = text.lower()
        problems += [f"blocklisted term in {f}" for t in blocked if t in low]

    emails = set(git("log", "--format=%ae%n%ce").split())
    if emails - {ALLOWED_EMAIL, "noreply@github.com"}:
        problems.append(f"commits with other identities: {sorted(emails - {ALLOWED_EMAIL})}")

    if problems:
        print("pre-push check FAILED; push blocked:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"pre-push check passed ({len(files)} tracked files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
