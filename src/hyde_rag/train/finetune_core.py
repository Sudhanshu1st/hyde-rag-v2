"""Domain adaptation F2: fine-tune bge on synthetic queries with mined hard negatives.

Self-contained (torch / transformers / sentence-transformers only): this file is embedded
verbatim into notebooks/finetune_food_embedder_colab.ipynb so the free-GPU run uses exactly the
code tested here. Never trains on eval questions: queries are generated from corpus chunks
by a different model (doc2query T5) with a different prompt than the eval set.
"""

from __future__ import annotations

import json
import random

import numpy as np
import torch

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOC2QUERY = "doc2query/msmarco-t5-base-v1"  # Apache-2.0


def index_text(title: str, section: str, text: str) -> str:
    """Same as DocChunk.index_text in the project: title and section path as a header."""
    header = f"{title} | {section}" if section else title
    return f"{header}\n{text}"


def load_food_chunks(chunks_path: str, docs_path: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """-> chunk_ids, index texts, raw texts, group keys (doc_id:page) for negative filtering."""
    titles = {}
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                titles[d["doc_id"]] = d["title"]
    ids, idx_texts, raw, groups = [], [], [], []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            ids.append(c["chunk_id"])
            idx_texts.append(index_text(titles[c["doc_id"]], c["section"], c["text"]))
            raw.append(c["text"])
            groups.append(f"{c['doc_id']}:{c['page']}")
    return ids, idx_texts, raw, groups


def generate_queries(
    texts: list[str],
    n_per_text: int = 3,
    model_id: str = DOC2QUERY,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 13,
    max_input_tokens: int = 384,
) -> list[list[str]]:
    """Sampled doc2query queries per text (top-p 0.95, top-k 10), deduplicated."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
    out: list[list[str]] = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            batch = tok(
                texts[s : s + batch_size],
                truncation=True,
                max_length=max_input_tokens,
                padding=True,
                return_tensors="pt",
            ).to(device)
            gen = model.generate(
                **batch, max_length=64, do_sample=True, top_p=0.95, top_k=10, num_return_sequences=n_per_text
            )
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            for i in range(len(dec) // n_per_text):
                qs = [q.strip() for q in dec[i * n_per_text : (i + 1) * n_per_text] if q.strip()]
                out.append(list(dict.fromkeys(qs)))
    return out


def encode(model, texts: list[str], prefix: str = "", batch_size: int = 64) -> np.ndarray:
    """Normalised embeddings from a SentenceTransformer (bge: CLS pooling + Normalize)."""
    return model.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def mine_negatives(
    query_vecs: np.ndarray,
    corpus_vecs: np.ndarray,
    positives: list[int],
    groups: list[str],
    rank_min: int = 10,
    rank_max: int = 50,
    seed: int = 13,
) -> list[int]:
    """One hard negative per query: a random corpus item ranked rank_min..rank_max for that query,
    excluding items from the positive's own page (likely false negatives)."""
    rng = random.Random(seed)
    q = torch.from_numpy(query_vecs)
    c = torch.from_numpy(corpus_vecs)
    negatives = []
    for start in range(0, len(q), 1024):
        top = torch.topk(q[start : start + 1024] @ c.T, k=min(rank_max, len(c)), dim=1).indices.numpy()
        for row, cand in enumerate(top):
            pos = positives[start + row]
            pool = [int(j) for j in cand[rank_min:rank_max] if groups[int(j)] != groups[pos]]
            negatives.append(rng.choice(pool) if pool else -1)
    return negatives


def train(
    base_model: str,
    anchors: list[str],
    positives: list[str],
    negatives: list[str],
    out_dir: str,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 2e-5,
    query_prefix: str = BGE_QUERY_PREFIX,
    fp16: bool = True,
    seed: int = 13,
) -> str:
    """MultipleNegativesRankingLoss on (anchor, positive, hard negative) triplets."""
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.base.sampler import BatchSamplers
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    model = SentenceTransformer(base_model)
    data = Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir + "-checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=0.1,  # a float is a ratio in transformers v5
        fp16=fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={"anchor": query_prefix},  # bge queries carry the instruction, passages don't
        seed=seed,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=data, loss=MultipleNegativesRankingLoss(model)
    )
    trainer.train()
    model.save(out_dir)
    return out_dir
