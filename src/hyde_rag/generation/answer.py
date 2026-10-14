"""Grounded answer generation with citations and a two-layer "no answer" fallback.

Layer (a), the retrieval gate: if the top reranker score is below tau (calibrated on the dev
split only), return the fixed abstention response without calling the LLM.
Layer (b), the prompt rule: the model must answer with the abstention text when the context
does not support an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hyde_rag.generation.citations import CitationReport, validate
from hyde_rag.generation.context import Chunk, format_context
from hyde_rag.llm.client import LLMRequest

ABSTAIN_TEXT = "Not found in the provided documents."

QA_SYSTEM = f"""You answer questions using only the numbered context passages provided.
Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- End every sentence with the ID(s) of the passage(s) that support it, in square brackets,
  e.g. [doc_a:3:1] or [doc_a:3:1, doc_b:7:0].
- Only cite IDs that appear in the context.
- If the context does not contain the answer, reply with exactly: {ABSTAIN_TEXT}
- Be concise: at most 5 sentences."""

CLAIM_SYSTEM = f"""You check a scientific claim against the provided abstracts only.
Reply in this format:
Verdict: SUPPORTED or Verdict: REFUTED
Then 1 to 3 sentences explaining why. End every sentence with the ID(s) of the abstract(s)
that support it, in square brackets, e.g. [12345:0:0].
Rules:
- Use only what the abstracts state. Do not use outside knowledge.
- Only cite IDs that appear in the context.
- If the abstracts neither support nor refute the claim, reply with exactly:
Verdict: NOT FOUND
{ABSTAIN_TEXT}"""

_VERDICT = re.compile(r"^\s*verdict\s*:\s*(supported|refuted|not found)\b", re.IGNORECASE | re.MULTILINE)


@dataclass
class Answer:
    text: str
    abstained: bool
    gated: bool = False  # abstained by the retrieval gate (no LLM call)
    verdict: str | None = None  # claim mode: SUPPORTED / REFUTED / NOT FOUND
    citations: CitationReport = field(default_factory=CitationReport)
    chunks: list[Chunk] = field(default_factory=list)


def build_request(
    question: str,
    chunks: list[Chunk],
    model: str,
    *,
    mode: str = "qa",
    reasoning_effort: str | None = None,
    max_tokens: int = 1024,
) -> LLMRequest:
    system = CLAIM_SYSTEM if mode == "claim" else QA_SYSTEM
    label = "Claim" if mode == "claim" else "Question"
    user = f"Context:\n{format_context(chunks)}\n\n{label}: {question}"
    return LLMRequest(
        model=model,
        messages=(("system", system), ("user", user)),
        max_tokens=max_tokens,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
    )


def gate(chunks: list[Chunk], tau: float | None) -> bool:
    """True if the retrieval gate says abstain (best score below tau)."""
    if tau is None or not chunks:
        return not chunks
    scores = [c.score for c in chunks if c.score is not None]
    return bool(scores) and max(scores) < tau


def gated_answer(chunks: list[Chunk], mode: str = "qa") -> Answer:
    text = f"Verdict: NOT FOUND\n{ABSTAIN_TEXT}" if mode == "claim" else ABSTAIN_TEXT
    return Answer(
        text=text, abstained=True, gated=True, verdict="NOT FOUND" if mode == "claim" else None, chunks=chunks
    )


def parse_answer(text: str, chunks: list[Chunk], mode: str = "qa") -> Answer:
    text = text.strip()
    verdict = None
    if mode == "claim":
        m = _VERDICT.search(text)
        verdict = m.group(1).upper() if m else None
    abstained = ABSTAIN_TEXT.lower() in text.lower() or verdict == "NOT FOUND"
    report = (
        CitationReport()
        if abstained
        else validate(text, {c.chunk_id for c in chunks}, exempt=("Verdict:", "verdict:"))
    )
    return Answer(text=text, abstained=abstained, verdict=verdict, citations=report, chunks=chunks)
