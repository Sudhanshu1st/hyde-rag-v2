"""Download SciFact claim labels (SUPPORT / CONTRADICT) from the original release.

BEIR ships SciFact for retrieval only. The original release (Wadden et al., 2020; CC BY-NC 2.0)
labels each claim's evidence, which lets us score answer correctness (the verdict) without
an LLM judge. Only the two claims files are extracted from the archive.

Usage:
    python scripts/download_scifact_labels.py
"""

from __future__ import annotations

import hashlib
import json
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
# No checksum is published upstream; pinned on first download (2026-10-14).
SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
WANTED = {"data/claims_dev.jsonl", "data/claims_train.jsonl"}
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "scifact_labels"


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def main() -> int:
    archive = OUT_DIR / "_downloads" / "data.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "hyde-rag-v2/0.1 (research)"})
        with urllib.request.urlopen(req, timeout=60, context=ssl_context()) as resp:
            archive.write_bytes(resp.read())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if SHA256 and digest != SHA256:
        print(f"sha256 mismatch: {digest}", file=sys.stderr)
        return 1
    print(f"sha256 {digest}")

    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name in WANTED and m.isfile()]
        if {m.name for m in members} != WANTED:
            print(f"archive is missing {WANTED - {m.name for m in members}}", file=sys.stderr)
            return 1
        for m in members:
            data = tar.extractfile(m).read()  # read bytes only; nothing executable is written
            (OUT_DIR / Path(m.name).name).write_bytes(data)

    for name in sorted(WANTED):
        path = OUT_DIR / Path(name).name
        claims = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        labelled = sum(1 for c in claims if c.get("evidence"))
        print(f"{path.name}: {len(claims)} claims, {labelled} with evidence labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
