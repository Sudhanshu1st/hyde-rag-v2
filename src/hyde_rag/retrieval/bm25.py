"""BM25 with bm25s, configured to approximate Pyserini's BEIR "flat" baseline.

Pyserini/Anserini (Lucene DefaultEnglishAnalyzer): StandardTokenizer, possessive filter,
lowercase, Lucene's 33 English stopwords, Porter stemmer; BM25 k1=0.9, b=0.4.
We mirror that with an explicit tokenizer and bm25s's "lucene" scoring variant.
Remaining differences are listed in docs/DECISIONS.md (D-RET-1).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import bm25s
import numpy as np
import Stemmer

# Lucene EnglishAnalyzer.ENGLISH_STOP_WORDS_SET (used by Anserini's DefaultEnglishAnalyzer).
LUCENE_ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such "  # noqa: SIM905
    "that the their then there these they this to was will with".split()
)

_POSSESSIVE = re.compile(r"['\N{RIGHT SINGLE QUOTATION MARK}]s\b")  # possessive 's
_TOKEN = re.compile(r"(?u)\b\w+\b")

Run = dict[str, dict[str, float]]  # query_id -> {doc_id: score}


class LuceneLikeAnalyzer:
    """lowercase -> strip possessives -> word tokens -> drop stopwords -> Porter stem."""

    def __init__(self, stopwords: Iterable[str] = LUCENE_ENGLISH_STOPWORDS, stemmer: str = "porter"):
        self.stopwords = frozenset(stopwords)
        self._stemmer = Stemmer.Stemmer(stemmer)

    def __call__(self, text: str) -> list[str]:
        text = _POSSESSIVE.sub("", text.lower())
        tokens = [t for t in _TOKEN.findall(text) if t not in self.stopwords]
        return self._stemmer.stemWords(tokens)


class BM25Retriever:
    def __init__(self, k1: float = 0.9, b: float = 0.4, method: str = "lucene"):
        self.k1, self.b, self.method = k1, b, method
        self.analyzer = LuceneLikeAnalyzer()
        self._model: bm25s.BM25 | None = None
        self._doc_ids: list[str] = []

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        self._doc_ids = list(doc_ids)
        self._model = bm25s.BM25(k1=self.k1, b=self.b, method=self.method)
        self._model.index([self.analyzer(t) for t in texts], show_progress=False)

    def search(self, queries: dict[str, str], k: int = 1000) -> Run:
        if self._model is None:
            raise RuntimeError("Call index() first.")
        vocab = self._model.vocab_dict
        qids = list(queries)
        # Unknown query terms can't match any document; drop them (an empty query scores 0).
        tokenized = [[t for t in self.analyzer(queries[q]) if t in vocab] for q in qids]
        k = min(k, len(self._doc_ids))
        run: Run = {}
        nonempty = [i for i, toks in enumerate(tokenized) if toks]
        if nonempty:
            idx, scores = self._model.retrieve(
                [tokenized[i] for i in nonempty], k=k, show_progress=False, n_threads=0
            )
            for row, i in enumerate(nonempty):
                run[qids[i]] = {
                    self._doc_ids[int(d)]: float(s)
                    for d, s in zip(idx[row], scores[row], strict=True)
                    if s > 0
                }
        for q in qids:
            run.setdefault(q, {})
        return run


def doc_text_flat(title: str, text: str) -> str:
    """Pyserini BeirFlatCollection: contents = title + "\\n" + text."""
    return f"{title}\n{text}" if title else text


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, sorted descending."""
    k = min(k, scores.shape[-1])
    part = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    order = np.take_along_axis(scores, part, axis=-1).argsort(axis=-1)[..., ::-1]
    return np.take_along_axis(part, order, axis=-1)
