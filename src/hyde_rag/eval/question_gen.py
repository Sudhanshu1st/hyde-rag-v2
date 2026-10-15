"""Draft the food-quality eval set: LLM-written candidate questions for human review.

* Chunks are sampled across documents (quota ~ sqrt(#chunks), so the 732-page Food Code
  doesn't dominate) and matched to question types by content (numbers with units for
  "numeric", must/shall/steps for "procedural", ...).
* Each call returns strict JSON with a verbatim evidence quote. The quote is located in the
  document's canonical text to give a gold char span; unlocated quotes are flagged for review.
* Multi-document questions use a chunk plus its nearest BM25 neighbour from another document.
* Unanswerable questions are plausible questions on a chunk's topic that it does not answer.
The question generator is a different model from the HyDE generator (configs/llm.yaml).
"""

from __future__ import annotations

import csv
import difflib
import itertools
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from hyde_rag import paths
from hyde_rag.ingest.chunking import DocChunk
from hyde_rag.ingest.food import load_chunks, load_docs, processed_dir
from hyde_rag.llm.client import LLMRequest
from hyde_rag.llm.runner import WorkItem

QUESTION_TYPES = ("factual", "numeric", "procedural", "jargon", "multi_doc", "unanswerable")
EVALSET_DIR = paths.REPO_ROOT / "eval_sets" / "food_quality"

# Candidate counts per type (about 230 candidates for ~200 reviewed items plus ~30 written by hand).
PLAN = {"factual": 50, "numeric": 50, "procedural": 40, "jargon": 30, "multi_doc": 20, "unanswerable": 40}

_NUMERIC = re.compile(
    r"\d+(?:\.\d+)?\s?(?:\N{DEGREE SIGN}\s?[FC]|degrees|hours?|minutes?|days?|ppm|percent|%|mg|log|seconds?)",
    re.IGNORECASE,
)
_PROCEDURAL = re.compile(r"\b(shall|must|should|steps?|procedures?|verify|monitor(?:ing)?|record)\b", re.I)
_WS = re.compile(r"\s+")

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "reference_answer": {"type": "string"},
                    "evidence_quotes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question", "reference_answer", "evidence_quotes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

SYSTEM = """You write evaluation questions for a search system over food-safety regulations and guidance
(FDA, USDA FSIS, Codex, FSSAI). Write questions the way a food-safety or quality manager would ask
them, without mentioning "the passage", "this document" or page numbers.
For every question give a short reference answer and copy 1-2 evidence quotes VERBATIM from the
passage(s): exact consecutive words, 8-40 words each, no ellipses."""

TYPE_INSTRUCTIONS = {
    "factual": "Write {n} factual question(s) answered by a specific statement in the passage.",
    "numeric": "Write {n} question(s) whose answer is a numeric limit stated in the passage "
    "(a temperature, time, concentration, frequency or similar), including its unit.",
    "procedural": "Write {n} question(s) asking how something must be done (steps, corrective actions, "
    "monitoring or verification) as described in the passage.",
    "jargon": "Write {n} very short keyword-style query (2-6 words, may be a fragment ending in '?', "
    "e.g. 'CCP for cooling?', 'Listeria swab frequency?') answered by the passage.",
    "multi_doc": "Write {n} question(s) that can only be answered by combining facts from BOTH passages. "
    "Give at least one evidence quote from each passage.",
    "unanswerable": "Write {n} realistic question(s) on the same topic that this passage does NOT answer "
    "(for example asking for a value, scope, frequency or requirement it never states). They must "
    "sound plausible to a quality manager. Set reference_answer to 'NOT ANSWERABLE'.",
}

# Unanswerable questions have no evidence, so their schema has no evidence field (an empty
# required array made the model emit malformed keys in the pilot).
UNANSWERABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "reference_answer": {"type": "string"}},
                "required": ["question", "reference_answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


@dataclass
class Candidate:
    id: str
    type: str
    question: str
    reference_answer: str
    answerable: bool
    evidence: list[dict]  # {doc_id, page, section, char_start, char_end, quote, located}
    source_chunks: list[str]


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


_CITATION = re.compile(r"\bet al\b|\(\s*(?:19|20)\d{2}[a-z]?\s*\)")


def eligible(chunk: DocChunk) -> bool:
    """Body text of reasonable length; skips link lists and bibliographies."""
    words = len(chunk.text.split())
    return 80 <= words <= 400 and chunk.text.count("http") <= 3 and len(_CITATION.findall(chunk.text)) < 3


def sample_chunks(chunks: list[DocChunk], seed: int = 13) -> dict[str, list[DocChunk]]:
    """Assign chunks to single-chunk question types, spread across documents."""
    rng = random.Random(seed)
    pool = [c for c in chunks if eligible(c)]
    by_doc: dict[str, list[DocChunk]] = {}
    for c in pool:
        by_doc.setdefault(c.doc_id, []).append(c)
    for v in by_doc.values():
        rng.shuffle(v)
    weights = {d: math.sqrt(len(v)) for d, v in by_doc.items()}

    def draw(n: int, accept) -> list[DocChunk]:
        """Round-robin over documents (weighted) until n accepted chunks are drawn."""
        picked: list[DocChunk] = []
        docs = sorted(by_doc, key=lambda d: -weights[d])
        credit = dict.fromkeys(docs, 0.0)
        while len(picked) < n and any(by_doc[d] for d in docs):
            for d in docs:
                credit[d] += weights[d]
            d = max((x for x in docs if by_doc[x]), key=lambda x: credit[x])
            credit[d] -= sum(weights.values())
            for i, c in enumerate(by_doc[d]):
                if accept(c):
                    picked.append(by_doc[d].pop(i))
                    break
            else:
                by_doc[d].clear()
        return picked

    plan = {
        "numeric": draw(PLAN["numeric"], lambda c: len(_NUMERIC.findall(c.text)) >= 2),
        "procedural": draw(PLAN["procedural"], lambda c: len(_PROCEDURAL.findall(c.text)) >= 3),
        "jargon": draw(PLAN["jargon"], lambda c: True),
        "factual": draw(PLAN["factual"], lambda c: True),
        "unanswerable": draw(PLAN["unanswerable"] // 2, lambda c: True),  # 2 questions per chunk
        "multi_doc_anchor": draw(PLAN["multi_doc"], lambda c: True),
    }
    return plan


def pair_for_multi_doc(
    anchors: list[DocChunk], chunks: list[DocChunk], titles: dict[str, str]
) -> list[tuple]:
    """Each anchor's most similar chunk from a different document (BM25 over chunk text)."""
    from hyde_rag.retrieval.bm25 import BM25Retriever

    pool = [c for c in chunks if eligible(c)]
    bm25 = BM25Retriever()
    bm25.index([c.chunk_id for c in pool], [c.index_text(titles[c.doc_id]) for c in pool])
    by_id = {c.chunk_id: c for c in pool}
    pairs = []
    run = bm25.search({a.chunk_id: a.text for a in anchors}, k=50)
    for a in anchors:
        for cid, _ in sorted(run[a.chunk_id].items(), key=lambda kv: -kv[1]):
            b = by_id[cid]
            if b.doc_id != a.doc_id:
                pairs.append((a, b))
                break
    return pairs


def build_items(
    plan: dict[str, list[DocChunk]],
    pairs: list[tuple],
    titles: dict[str, str],
    model: str,
    reasoning: str | None,
) -> list[tuple[WorkItem, str, list[DocChunk]]]:
    items = []

    def passage(c: DocChunk) -> str:
        return f"Source: {titles[c.doc_id]} | {c.section}\n{c.text}"

    def make(qtype: str, n: int, chunks: list[DocChunk]) -> None:
        body = "\n\n".join(f"Passage {i + 1}:\n{passage(c)}" for i, c in enumerate(chunks))
        user = f"{body}\n\nTask: {TYPE_INSTRUCTIONS[qtype].format(n=n)}"
        req = LLMRequest(
            model=model,
            messages=(("system", SYSTEM), ("user", user)),
            max_tokens=1500,
            temperature=0.7,
            reasoning_effort=reasoning,
            json_schema=UNANSWERABLE_SCHEMA if qtype == "unanswerable" else QUESTION_SCHEMA,
        )
        item_id = f"{qtype}|" + "+".join(c.chunk_id for c in chunks)
        items.append((WorkItem(item_id, 0, req), qtype, chunks))

    for qtype in ("factual", "numeric", "procedural", "jargon"):
        for c in plan[qtype]:
            make(qtype, 1, [c])
    for c in plan["unanswerable"]:
        make("unanswerable", 2, [c])
    for a, b in pairs:
        make("multi_doc", 1, [a, b])
    return items


_ELLIPSIS = re.compile(r"\.\.\.|\N{HORIZONTAL ELLIPSIS}")


def locate(quote: str, chunk: DocChunk, doc_text: str) -> tuple[int, int] | None:
    """Char span of a quote in the document, searched within its source chunk.

    A quote shortened with "..." is located fragment by fragment; the span runs from the first
    fragment to the last (all must be found, in order).
    """
    fragments = [f for f in (_norm(p).strip(" .\"'") for p in _ELLIPSIS.split(quote)) if len(f) >= 10]
    if len(fragments) <= 1:
        return _locate_one(_norm(quote).strip(" .\"'"), chunk, doc_text)
    spans = [_locate_one(f, chunk, doc_text) for f in fragments]
    if any(s is None for s in spans) or any(a[1] > b[0] for a, b in itertools.pairwise(spans)):
        return None
    return spans[0][0], spans[-1][1]


def _locate_one(q: str, chunk: DocChunk, doc_text: str) -> tuple[int, int] | None:
    """Exact match after whitespace/case normalisation, else the best fuzzy match of the same
    length if it is at least 90% similar."""
    if len(q) < 10:
        return None
    text = doc_text[chunk.char_start : chunk.char_end]
    # Map normalised positions back to raw offsets.
    raw_idx, norm_chars = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space or not norm_chars:
                continue
            norm_chars.append(" ")
            raw_idx.append(i)
            prev_space = True
        else:
            norm_chars.append(ch.lower())
            raw_idx.append(i)
            prev_space = False
    norm = "".join(norm_chars)
    pos = norm.find(q)
    if pos < 0:
        sm = difflib.SequenceMatcher(None, norm, q, autojunk=False)
        best = sm.find_longest_match(0, len(norm), 0, len(q))
        if best.size < 0.6 * len(q):
            return None
        start = max(0, best.a - best.b)
        window = norm[start : start + len(q)]
        if difflib.SequenceMatcher(None, window, q).ratio() < 0.9:
            return None
        pos = start
    end = min(pos + len(q), len(raw_idx)) - 1
    return chunk.char_start + raw_idx[pos], chunk.char_start + raw_idx[end] + 1


def parse_candidates(
    raw: str, item_id: str, qtype: str, chunks: list[DocChunk], doc_texts: dict[str, str]
) -> list[Candidate]:
    data = json.loads(raw)
    out = []
    for k, q in enumerate(data.get("questions", [])):
        evidence = []
        if qtype != "unanswerable":
            for quote in q.get("evidence_quotes", []):
                hit = None
                for c in chunks:
                    span = locate(quote, c, doc_texts[c.doc_id])
                    if span:
                        hit = {
                            "doc_id": c.doc_id,
                            "page": c.page,
                            "section": c.section,
                            "char_start": span[0],
                            "char_end": span[1],
                            "quote": quote,
                            "located": True,
                        }
                        break
                evidence.append(
                    hit
                    or {
                        "doc_id": chunks[0].doc_id,
                        "page": chunks[0].page,
                        "section": chunks[0].section,
                        "char_start": None,
                        "char_end": None,
                        "quote": quote,
                        "located": False,
                    }
                )
        out.append(
            Candidate(
                id=f"{qtype[:3]}-{abs(hash(item_id)) % 10**6:06d}-{k}",
                type=qtype,
                question=q["question"].strip(),
                reference_answer=q["reference_answer"].strip(),
                answerable=qtype != "unanswerable",
                evidence=evidence,
                source_chunks=[c.chunk_id for c in chunks],
            )
        )
    return out


REVIEW_COLUMNS = [
    "id",
    "type",
    "answerable",
    "question",
    "reference_answer",
    "evidence_doc",
    "evidence_page",
    "evidence_quote",
    "evidence_located",
    "decision",
    "edited_question",
    "edited_answer",
    "notes",
]


def write_review_csv(candidates: list[Candidate], path: Path, n_blank_human: int = 30) -> None:
    """One row per candidate. decision: accept | edit | reject (blank = not reviewed).
    Blank 'human' rows at the end are for questions you write yourself."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel opens UTF-8 correctly
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for c in candidates:
            ev = c.evidence[0] if c.evidence else {}
            w.writerow(
                {
                    "id": c.id,
                    "type": c.type,
                    "answerable": c.answerable,
                    "question": c.question,
                    "reference_answer": c.reference_answer,
                    "evidence_doc": ev.get("doc_id", ""),
                    "evidence_page": ev.get("page", ""),
                    "evidence_quote": " || ".join(e["quote"] for e in c.evidence),
                    "evidence_located": all(e["located"] for e in c.evidence) if c.evidence else "",
                    "decision": "",
                    "edited_question": "",
                    "edited_answer": "",
                    "notes": "",
                }
            )
        for i in range(n_blank_human):
            w.writerow(
                {
                    "id": f"hum-{i + 1:03d}",
                    "type": "human",
                    "answerable": "",
                    "question": "",
                    "reference_answer": "",
                    "evidence_doc": "",
                    "evidence_page": "",
                    "evidence_quote": "",
                    "evidence_located": "",
                    "decision": "",
                    "edited_question": "",
                    "edited_answer": "",
                    "notes": "write your own question; give evidence_doc and a verbatim evidence_quote",
                }
            )


def load_inputs() -> tuple[list[DocChunk], dict[str, str], dict[str, str]]:
    chunks = load_chunks()
    docs = load_docs()
    titles = {d: v["title"] for d, v in docs.items()}
    texts = {d: (processed_dir() / "text" / f"{d}.txt").read_text(encoding="utf-8") for d in docs}
    return chunks, titles, texts


def generate(
    *, llm: bool = True, approved: bool = False, pilot_only: bool = False, seed: int = 13, log=print
) -> Path:
    """Draft candidates with the question-generator model and write the review CSV."""
    import yaml

    from hyde_rag.llm.cache import JsonlCache, cache_key
    from hyde_rag.llm.client import default_base_url
    from hyde_rag.llm.quota import Limits, load_limits
    from hyde_rag.llm.runner import PILOT_CALLS, run_bulk

    llm_cfg = yaml.safe_load((paths.CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8"))
    model = llm_cfg["roles"]["question_gen"]
    if model == llm_cfg["roles"]["hyde_main"]:
        raise ValueError("The question generator must differ from the HyDE generator.")
    reasoning = llm_cfg.get("generation", {}).get(model, {}).get("reasoning_effort")

    chunks, titles, texts = load_inputs()
    plan = sample_chunks(chunks, seed=seed)
    pairs = pair_for_multi_doc(plan.pop("multi_doc_anchor"), chunks, titles)
    items = build_items(plan, pairs, titles, model, reasoning)
    random.Random(seed).shuffle(items)  # so a 20-call pilot covers every question type
    counts = {t: sum(1 for _, qt, _ in items if qt == t) for t in QUESTION_TYPES}
    log(f"{len(items)} generation calls with {model}: {counts}")

    cache = JsonlCache(paths.cache_dir() / "food" / "question_gen" / f"{model.replace('/', '__')}.jsonl")
    if llm:
        from hyde_rag.llm.client import OpenAICompatClient

        client = OpenAICompatClient()
        limits = load_limits().get(model, Limits(10**6, 10**9, 10**12, 10**15))
        run_bulk(
            client,
            cache,
            [it for it, _, _ in items],
            limits,
            approved=approved,
            limit=PILOT_CALLS if pilot_only else None,
            log=log,
        )
        base_url = client.base_url
    else:
        base_url = default_base_url()

    candidates: list[Candidate] = []
    bad = 0
    for it, qtype, cs in items:
        rec = cache.get(cache_key(base_url, it.request, 0))
        if rec is None:
            continue
        try:
            candidates.extend(parse_candidates(rec.text, it.item_id, qtype, cs, texts))
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
    located = sum(1 for c in candidates if c.evidence and all(e["located"] for e in c.evidence))
    log(
        f"{len(candidates)} candidates from {len(items)} calls ({bad} unparseable); "
        f"evidence located for {located}/{sum(1 for c in candidates if c.answerable)} answerable"
    )

    EVALSET_DIR.mkdir(parents=True, exist_ok=True)
    with (EVALSET_DIR / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c.__dict__, ensure_ascii=False))
            f.write("\n")
    review = EVALSET_DIR / "review.csv"
    write_review_csv(candidates, review)
    return review
