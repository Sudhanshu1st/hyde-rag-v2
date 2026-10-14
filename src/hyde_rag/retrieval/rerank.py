"""Cross-encoder reranking with a persistent (query_id, doc_id) score cache (D-RET-7).

Systems that share candidates share scores, so reranking many systems costs little more
than reranking the union of their top-100 lists.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderReranker:
    def __init__(
        self, model_id: str = DEFAULT_RERANKER, cache_file: Path | None = None, threads: int | None = None
    ):
        import torch
        from sentence_transformers import CrossEncoder

        if threads:
            torch.set_num_threads(threads)
        self.model_id = model_id
        self._model = CrossEncoder(model_id, device="cpu")
        self._cache_file = cache_file
        self._scores: dict[str, float] = {}
        if cache_file and cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                if line:
                    key, score = line.rsplit("\t", 1)
                    self._scores[key] = float(score)

    @staticmethod
    def _key(qid: str, did: str) -> str:
        return f"{qid}\x1f{did}"

    def score(self, pairs: list[tuple[str, str, str, str]], batch_size: int = 32) -> list[float]:
        """pairs: (query_id, doc_id, query_text, doc_text) -> scores, using the cache."""
        missing = [p for p in pairs if self._key(p[0], p[1]) not in self._scores]
        if missing:
            # Sort by length to reduce padding; restore via the cache dict.
            missing.sort(key=lambda p: len(p[2]) + len(p[3]))
            values = self._model.predict(
                [(p[2], p[3]) for p in missing], batch_size=batch_size, show_progress_bar=False
            )
            new_lines = []
            for p, v in zip(missing, values, strict=True):
                k = self._key(p[0], p[1])
                self._scores[k] = float(v)
                new_lines.append(f"{k}\t{float(v)!r}")
            if self._cache_file:
                self._cache_file.parent.mkdir(parents=True, exist_ok=True)
                with self._cache_file.open("a", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
        return [self._scores[self._key(p[0], p[1])] for p in pairs]


def cache_file_for(index_dir: Path, model_id: str) -> Path:
    return index_dir / f"rerank__{model_id.replace('/', '__')}.tsv"
