"""Download BEIR datasets (official UKP mirror) with checksum verification.

Usage:
    python scripts/download_beir.py                 # scifact trec-covid arguana
    python scripts/download_beir.py scifact         # just one

Each archive is verified against the md5 published in the BEIR README, then
extracted into its own directory under data/beir/<name>/. Archive members are
validated before extraction (no absolute paths, no "..", data files only).
Dataset licences stay with the original authors; see NOTICE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# md5 checksums as published in the BEIR README (https://github.com/beir-cellar/beir).
MD5 = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "trec-covid": "ce62140cb23feb9becf6270d0d1fe6d1",
    "arguana": "8ad3e3c2a5867cdced806d6503f29b99",
}

ALLOWED_SUFFIXES = {".jsonl", ".tsv", ""}  # "" = directory entries
REPO_ROOT = Path(__file__).resolve().parents[1]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ssl_context() -> ssl.SSLContext:
    """Verify against certifi's CA bundle when available.

    On Windows, Python reads the OS certificate store, which may not have
    lazily installed newer roots (e.g. HARICA, used by the UKP server), so
    verification fails even though the chain is valid. certifi ships them.
    Verification is never disabled.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hyde-rag-v2/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=60, context=ssl_context()) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def safe_extract(zip_path: Path, target: Path) -> None:
    """Extract only expected data files, refusing path traversal."""
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            p = PurePosixPath(member.filename)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            if p.suffix not in ALLOWED_SUFFIXES:
                raise ValueError(f"Unexpected file type in archive: {member.filename}")
        # Archives contain a single top-level folder named after the dataset.
        zf.extractall(target)


def summarize(ds_dir: Path) -> dict:
    def count_lines(p: Path) -> int:
        with p.open(encoding="utf-8") as f:
            return sum(1 for _ in f)

    stats: dict = {
        "corpus": count_lines(ds_dir / "corpus.jsonl"),
        "queries": count_lines(ds_dir / "queries.jsonl"),
    }
    for split_file in sorted((ds_dir / "qrels").glob("*.tsv")):
        qids = set()
        with split_file.open(encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                qids.add(line.split("\t", 1)[0])
        stats[f"qrels_{split_file.stem}_queries"] = len(qids)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("datasets", nargs="*", default=list(MD5), choices=list(MD5))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "beir")
    args = parser.parse_args()

    manifest = {}
    for name in args.datasets:
        zip_path = args.out / "_downloads" / f"{name}.zip"
        ds_dir = args.out / name
        if not zip_path.exists():
            print(f"[{name}] downloading {BASE_URL.format(name=name)}")
            download(BASE_URL.format(name=name), zip_path)
        digest = md5sum(zip_path)
        if digest != MD5[name]:
            print(f"[{name}] md5 mismatch: got {digest}, expected {MD5[name]}", file=sys.stderr)
            return 1
        print(f"[{name}] md5 ok")
        if not (ds_dir / "corpus.jsonl").exists():
            safe_extract(zip_path, args.out)
        stats = summarize(ds_dir)
        manifest[name] = {
            "url": BASE_URL.format(name=name),
            "md5": digest,
            "sha256": sha256sum(zip_path),
            **stats,
        }
        print(f"[{name}] {json.dumps(stats)}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
