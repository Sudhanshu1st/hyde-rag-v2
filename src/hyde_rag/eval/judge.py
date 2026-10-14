"""Claim-level LLM judge with strict JSON-schema output (D-EVAL-5).

* faithfulness: each answer sentence is split into atomic claims, and each claim is judged
  against the passages that sentence cites (not the whole context).
* citation precision: for each (sentence, cited passage) pair, does the passage support it?
* correctness: answer vs a reference answer (food set). SciFact uses gold labels instead.

The judge must be a different model from the answerer (configs/llm.yaml), and its numbers are
only trusted after agreement with 50 hand labels is reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from hyde_rag.generation.answer import Answer
from hyde_rag.llm.client import LLMRequest

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "integer"},
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["supported", "not_supported"]},
                },
                "required": ["sentence", "claim", "verdict"],
                "additionalProperties": False,
            },
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "integer"},
                    "passage_id": {"type": "string"},
                    "supports": {"type": "boolean"},
                },
                "required": ["sentence", "passage_id", "supports"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims", "citations"],
    "additionalProperties": False,
}

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

FAITHFULNESS_SYSTEM = """You are a strict fact-checking judge. You receive an answer split into numbered
sentences. Each sentence lists the passages it cites, with their text.
Tasks:
1. Split every sentence into atomic claims (one fact each). For each claim, verdict "supported"
   only if the passages cited BY THAT SENTENCE state it (paraphrase is fine). Anything that needs
   outside knowledge, goes beyond the passages, or contradicts them is "not_supported".
2. For every (sentence, cited passage) pair, say whether that passage supports the sentence.
Use the sentence numbers and passage IDs exactly as given. Judge only against the passages."""

CORRECTNESS_SYSTEM = """You grade an answer against a reference answer for the same question.
"correct": states the reference's key facts (numbers, limits, steps) without contradicting it.
"partially_correct": some key facts right, others missing or wrong.
"incorrect": wrong, contradicts the reference, or abstains when the reference has an answer.
Give a one-sentence reason."""


def faithfulness_request(answer: Answer, model: str, reasoning_effort: str | None = None) -> LLMRequest:
    by_id = {c.chunk_id: c for c in answer.chunks}
    lines = []
    for i, sentence in enumerate(answer.citations.sentences):
        lines.append(f"Sentence {i}: {sentence.text}")
        for cid in sentence.cited_ids:
            chunk = by_id.get(cid)
            if chunk is not None:
                lines.append(f"  Passage {cid}: {chunk.text.strip()}")
        if not sentence.cited_ids:
            lines.append("  (no passages cited)")
    return LLMRequest(
        model=model,
        messages=(("system", FAITHFULNESS_SYSTEM), ("user", "\n".join(lines))),
        max_tokens=2048,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
        json_schema=FAITHFULNESS_SCHEMA,
    )


def correctness_request(
    question: str, answer_text: str, reference: str, model: str, reasoning_effort: str | None = None
) -> LLMRequest:
    user = f"Question: {question}\nReference answer: {reference}\nAnswer to grade: {answer_text}"
    return LLMRequest(
        model=model,
        messages=(("system", CORRECTNESS_SYSTEM), ("user", user)),
        max_tokens=512,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
        json_schema=CORRECTNESS_SCHEMA,
    )


@dataclass(frozen=True)
class FaithfulnessResult:
    n_claims: int
    n_supported: int
    n_citations: int
    n_supporting_citations: int

    @property
    def faithfulness(self) -> float | None:
        return self.n_supported / self.n_claims if self.n_claims else None

    @property
    def citation_precision(self) -> float | None:
        return self.n_supporting_citations / self.n_citations if self.n_citations else None


def parse_faithfulness(raw: str, answer: Answer) -> FaithfulnessResult:
    """Parse judge JSON; citation pairs are restricted to those the answer actually made."""
    data = json.loads(raw)
    claims = data.get("claims", [])
    made = {(i, cid) for i, s in enumerate(answer.citations.sentences) for cid in s.cited_ids}
    judged = {
        (int(c["sentence"]), str(c["passage_id"])): bool(c["supports"]) for c in data.get("citations", [])
    }
    # Pairs the judge skipped count as not supporting (conservative).
    supporting = sum(1 for pair in made if judged.get(pair, False))
    return FaithfulnessResult(
        n_claims=len(claims),
        n_supported=sum(1 for c in claims if c.get("verdict") == "supported"),
        n_citations=len(made),
        n_supporting_citations=supporting,
    )


def parse_correctness(raw: str) -> str:
    return json.loads(raw)["verdict"]
