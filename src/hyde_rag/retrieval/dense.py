"""Dense retrieval: encoders, resumable corpus encoding and exact inner-product search.

Both encoders run through plain `transformers` so pooling is explicit:
  * Contriever (paper parity, D-RET-2): mean pooling, no normalisation, max_length 512,
    documents as title + " " + text.
  * bge-*-en-v1.5 (D-RET-3): CLS pooling + L2 normalisation, query instruction prefix,
    no passage prefix.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hyde_rag import paths

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    model_id: str
    pooling: str  # "mean" or "cls"
    normalize: bool
    query_prefix: str = ""
    max_length: int = 512


ENCODERS: dict[str, EncoderSpec] = {
    "contriever": EncoderSpec("contriever", "facebook/contriever", pooling="mean", normalize=False),
    "bge-base": EncoderSpec(
        "bge-base", "BAAI/bge-base-en-v1.5", pooling="cls", normalize=True, query_prefix=BGE_QUERY_PREFIX
    ),
    "bge-small": EncoderSpec(
        "bge-small", "BAAI/bge-small-en-v1.5", pooling="cls", normalize=True, query_prefix=BGE_QUERY_PREFIX
    ),
}


def doc_text_dense(title: str, text: str) -> str:
    """Contriever's BEIR convention (beir_utils.py): title + " " + text."""
    return f"{title} {text}".strip() if title else text


class Encoder:
    def __init__(self, spec: EncoderSpec, device: str = "cpu", threads: int | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.spec = spec
        self.device = device
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModel.from_pretrained(spec.model_id).to(device).eval()

    def encode(self, texts: list[str], *, is_query: bool, batch_size: int = 16) -> np.ndarray:
        """Encode texts; returns float32 [n, dim]. Batches are length-sorted to cut padding."""
        torch = self._torch
        if is_query and self.spec.query_prefix:
            texts = [self.spec.query_prefix + t for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                idx = order[start : start + batch_size]
                batch = self.tokenizer(
                    [texts[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                if self.spec.pooling == "mean":
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    emb = hidden[:, 0]
                if self.spec.normalize:
                    emb = torch.nn.functional.normalize(emb, dim=-1)
                out[idx] = emb.float().cpu().numpy()
        return out


@dataclass
class DenseIndex:
    doc_ids: list[str]
    embeddings: np.ndarray  # float32 (exact parity; TREC-COVID is ~526 MB)

    def search(self, query_vecs: np.ndarray, k: int = 1000, chunk: int = 64) -> list[list[tuple[str, float]]]:
        from hyde_rag.retrieval.bm25 import top_k

        results = []
        for start in range(0, len(query_vecs), chunk):
            scores = query_vecs[start : start + chunk] @ self.embeddings.T
            idx = top_k(scores, k)
            for row in range(scores.shape[0]):
                results.append([(self.doc_ids[j], float(scores[row, j])) for j in idx[row]])
        return results


def index_path(dataset: str, encoder: str) -> Path:
    return paths.index_dir() / dataset / encoder


def encode_corpus(
    encoder: Encoder,
    doc_ids: list[str],
    texts: list[str],
    out_dir: Path,
    *,
    shard_size: int = 1024,
    batch_size: int = 16,
    progress: bool = True,
) -> DenseIndex:
    """Encode a corpus into shards so an interrupted run resumes where it stopped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    n_shards = (len(texts) + shard_size - 1) // shard_size
    for s in range(n_shards):
        shard_file = shard_dir / f"{s:05d}.npy"
        if shard_file.exists():
            continue
        lo, hi = s * shard_size, min((s + 1) * shard_size, len(texts))
        emb = encoder.encode(texts[lo:hi], is_query=False, batch_size=batch_size)
        tmp = shard_dir / f"{s:05d}.tmp.npy"
        np.save(tmp, emb.astype(np.float32))
        tmp.replace(shard_file)
        if progress:
            print(f"  [{encoder.spec.name}] shard {s + 1}/{n_shards} ({hi}/{len(texts)} docs)", flush=True)

    embeddings = np.concatenate([np.load(shard_dir / f"{s:05d}.npy") for s in range(n_shards)])
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "doc_ids.json").write_text(json.dumps(doc_ids), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(asdict(encoder.spec), indent=2), encoding="utf-8")
    return DenseIndex(doc_ids, embeddings.astype(np.float32))


def load_index(dataset: str, encoder: str) -> DenseIndex | None:
    path = index_path(dataset, encoder)
    if not (path / "embeddings.npy").exists():
        return None
    doc_ids = json.loads((path / "doc_ids.json").read_text(encoding="utf-8"))
    return DenseIndex(doc_ids, np.load(path / "embeddings.npy").astype(np.float32))
