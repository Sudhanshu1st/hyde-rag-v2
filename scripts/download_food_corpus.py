"""Download the public food-safety corpus listed in data_sources.yaml, politely.

* fda.gov asks for 30 s between requests (robots.txt Crawl-Delay) and runs bot detection:
  one request at a time, and if a host blocks us we stop asking it (never evade).
* Public-domain files (commit_raw: true) go to data/food/raw/ and are committed to git;
  everything else goes to data/food/raw_restricted/ (gitignored) and is fetched by this script.
* Each file is checked to be a real PDF, then its sha256, size and retrieval date are written
  back into data_sources.yaml. Re-runs verify existing files against the recorded sha256.
* Hosts that refuse scripted downloads (e.g. usda.gov from some networks) get a manual
  fallback: save the file from a browser to the printed path and re-run to record its hash.

Usage:
    python scripts/download_food_corpus.py              # everything not yet downloaded
    python scripts/download_food_corpus.py --only fda_pchf_ch01 fda_pchf_ch02
    python scripts/download_food_corpus.py --verify     # check files against data_sources.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = REPO_ROOT / "data_sources.yaml"
RAW = REPO_ROOT / "data" / "food" / "raw"
RESTRICTED = REPO_ROOT / "data" / "food" / "raw_restricted"
USER_AGENT = "hyde-rag-v2/0.1 (open research benchmark; github.com/Sudhanshu1st)"
HOST_DELAY_S = {"www.fda.gov": 30, "openknowledge.fao.org": 10}
DEFAULT_DELAY_S = 5
MAX_ATTEMPTS = 3


class Blocked(RuntimeError):
    """The host refused a scripted download (403, or a bot-detection page)."""


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def target_path(doc: dict) -> Path:
    ext = {"pdf": ".pdf", "html": ".html"}.get(doc.get("format", "pdf"), ".bin")
    return (RAW if doc["commit_raw"] else RESTRICTED) / f"{doc['doc_id']}{ext}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, expect_pdf: bool) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=120, context=ssl_context()) as resp:
            final_url, ctype, body = resp.geturl(), resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 429):
            raise Blocked(f"HTTP {e.code}") from e
        raise
    if "apology" in final_url or "abuse" in final_url:
        raise Blocked(f"redirected to {final_url}")
    if expect_pdf and not body.startswith(b"%PDF-"):
        raise Blocked(f"expected a PDF, got {ctype or 'unknown type'} ({len(body)} bytes)")
    return body


def record(text: str, doc_id: str, fields: dict) -> str:
    """Insert/replace retrieved/sha256/bytes right after `doc_id:` in the entry's first line."""
    pattern = re.compile(
        rf"\{{doc_id: {re.escape(doc_id)},( retrieved: \"[^\"]*\", sha256: \"[^\"]*\", bytes: \d+,)?"
    )
    meta = f' retrieved: "{fields["retrieved"]}", sha256: "{fields["sha256"]}", bytes: {fields["bytes"]},'
    new, n = pattern.subn(f"{{doc_id: {doc_id},{meta}", text, count=1)
    if n != 1:
        raise ValueError(f"could not find entry for {doc_id} in {SOURCES.name}")
    return new


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", help="doc_ids to process")
    parser.add_argument("--verify", action="store_true", help="only verify existing files")
    args = parser.parse_args()

    text = SOURCES.read_text(encoding="utf-8")
    docs = yaml.safe_load(text)["documents"]
    if args.only:
        docs = [d for d in docs if d["doc_id"] in set(args.only)]
    RAW.mkdir(parents=True, exist_ok=True)
    RESTRICTED.mkdir(parents=True, exist_ok=True)

    last_request: dict[str, float] = {}
    blocked_hosts: dict[str, str] = {}
    downloaded, verified, manual, skipped, failed = [], [], [], [], []
    for doc in docs:
        doc_id, url, path = doc["doc_id"], doc.get("url"), target_path(doc)
        if doc.get("excluded"):
            continue
        if path.exists():
            digest = sha256(path)
            if doc.get("sha256") and doc["sha256"] != digest:
                print(f"[{doc_id}] SHA256 MISMATCH: file differs from data_sources.yaml", file=sys.stderr)
                failed.append(doc_id)
                continue
            if not doc.get("sha256"):  # e.g. a manually downloaded file: record it now
                fields = {
                    "retrieved": date.today().isoformat(),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
                text = record(text, doc_id, fields)
            verified.append(doc_id)
            continue
        if args.verify:
            print(f"[{doc_id}] missing: {path.relative_to(REPO_ROOT)}")
            failed.append(doc_id)
            continue
        if not url:
            print(f"[{doc_id}] no URL yet; skipped")
            skipped.append(doc_id)
            continue
        host = urlparse(url).netloc
        if host in blocked_hosts:
            manual.append((doc_id, url, path))
            continue

        delay = HOST_DELAY_S.get(host, DEFAULT_DELAY_S)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            wait = last_request.get(host, 0) + delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last_request[host] = time.monotonic()
            try:
                body = fetch(url, expect_pdf=doc.get("format", "pdf") == "pdf")
            except Blocked as e:
                print(f"[{doc_id}] {host} refused a scripted download ({e}); not retrying this host")
                blocked_hosts[host] = str(e)
                manual.append((doc_id, url, path))
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                print(f"[{doc_id}] attempt {attempt} failed: {e}")
                if attempt == MAX_ATTEMPTS:
                    failed.append(doc_id)
                else:
                    time.sleep(60 * attempt)
                continue
            path.write_bytes(body)
            fields = {"retrieved": date.today().isoformat(), "sha256": sha256(path), "bytes": len(body)}
            text = record(text, doc_id, fields)
            SOURCES.write_text(text, encoding="utf-8")  # save progress after every file
            downloaded.append(doc_id)
            print(f"[{doc_id}] {len(body) / 1e6:.1f} MB -> {path.relative_to(REPO_ROOT)}", flush=True)
            break

    SOURCES.write_text(text, encoding="utf-8")
    print(
        f"\ndownloaded {len(downloaded)}, already present {len(verified)}, manual {len(manual)}, "
        f"skipped {len(skipped)}, failed {len(failed)}"
    )
    if manual:
        print("\nThese hosts refuse scripted downloads. Open each URL in a browser, save the file to the")
        print("path shown, then re-run this script to verify and record it:")
        for doc_id, url, path in manual:
            print(f"  {doc_id}\n    {url}\n    -> {path.relative_to(REPO_ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
