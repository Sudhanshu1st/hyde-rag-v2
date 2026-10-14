"""Append-only JSONL cache for LLM outputs (D-LLM-3).

Key = sha256 of (base URL, canonical request parameters, sample index). Records store the
item id and the output, never the prompt, so committed BEIR caches contain no dataset text.
Re-runs read from here and cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hyde_rag.llm.client import LLMRequest, LLMResponse


def cache_key(base_url: str, request: LLMRequest, sample: int) -> str:
    payload = {"base_url": base_url.rstrip("/"), "request": request.params(), "sample": sample}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedOutput:
    key: str
    item_id: str
    sample: int
    text: str
    finish_reason: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    created: str


class JsonlCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: dict[str, CachedOutput] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = CachedOutput(**json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        continue  # a torn final line from an interrupted write
                    self._records[rec.key] = rec

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: str) -> CachedOutput | None:
        return self._records.get(key)

    def records(self) -> list[CachedOutput]:
        return list(self._records.values())

    def put(self, key: str, item_id: str, sample: int, response: LLMResponse) -> CachedOutput:
        rec = CachedOutput(
            key=key,
            item_id=item_id,
            sample=sample,
            text=response.text,
            finish_reason=response.finish_reason,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_s=round(response.latency_s, 3),
            created=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_asdict(rec), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._records[key] = rec
        return rec


def _asdict(rec: CachedOutput) -> dict[str, Any]:
    return dict(rec.__dict__)
