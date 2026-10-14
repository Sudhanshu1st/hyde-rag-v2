"""HyDE generation requests and multi-query rewrites (paper Sec. 3, Appendix A.1).

The prompts are built here; the free-tier runner (llm/runner.py) executes and caches them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from hyde_rag import paths
from hyde_rag.llm.client import LLMRequest
from hyde_rag.llm.runner import WorkItem

INSTRUCTIONS_PATH = paths.CONFIG_DIR / "hyde_instructions.yaml"
LLM_CONFIG_PATH = paths.CONFIG_DIR / "llm.yaml"

# Paper / official repo sampling settings (D-HYDE-3).
TEMPERATURE = 0.7
TOP_P = 1.0
STOP = ("\n\n\n",)
PASSAGE_TOKENS = 512
REASONING_HEADROOM = 512  # gpt-oss counts its (low-effort) reasoning inside the completion budget

MULTI_QUERY_K = 4
MULTI_QUERY_PROMPT = (
    "Write {k} different search queries that could find documents relevant to the query below. "
    "Vary the wording and the angle. Output one query per line, with no numbering or extra text.\n"
    "Query: {query}"
)


@dataclass(frozen=True)
class Instruction:
    id: str  # e.g. "paper.scifact" or "domain.food_quality_v1"
    template: str


def load_instruction(instruction_id: str, path: Path = INSTRUCTIONS_PATH) -> Instruction:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    group, name = instruction_id.split(".", 1)
    return Instruction(instruction_id, raw[group][name])


def system_prompt(path: Path = INSTRUCTIONS_PATH) -> str:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["system_passage_only"]


def generation_settings(model: str, path: Path = LLM_CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("generation", {}).get(model, {})


def hyde_items(
    queries: dict[str, str], instruction: Instruction, model: str, n_samples: int
) -> list[WorkItem]:
    """N independent samples per query, at the paper's sampling settings."""
    settings = generation_settings(model)
    reasoning = settings.get("reasoning_effort")
    max_tokens = PASSAGE_TOKENS + (REASONING_HEADROOM if reasoning not in (None, "none") else 0)
    system = system_prompt()
    items = []
    for qid, query in queries.items():
        user = instruction.template.format(query=query)
        for s in range(n_samples):
            request = LLMRequest(
                model=model,
                messages=(("system", system), ("user", user)),
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=STOP,
                seed=s,
                reasoning_effort=reasoning,
            )
            items.append(WorkItem(qid, s, request))
    return items


def multi_query_items(queries: dict[str, str], model: str, k: int = MULTI_QUERY_K) -> list[WorkItem]:
    settings = generation_settings(model)
    reasoning = settings.get("reasoning_effort")
    max_tokens = 256 + (REASONING_HEADROOM if reasoning not in (None, "none") else 0)
    return [
        WorkItem(
            qid,
            0,
            LLMRequest(
                model=model,
                messages=(("user", MULTI_QUERY_PROMPT.format(k=k, query=query)),),
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
                reasoning_effort=reasoning,
            ),
        )
        for qid, query in queries.items()
    ]


_NUMBERING = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*\*|__|`)")
_WS = re.compile(r"\s+")


def clean_passage(text: str) -> str:
    """Strip markdown markup (headings, bold, code ticks) and collapse whitespace (D-HYDE-7).

    Applied identically to every generator's output before encoding; the cache keeps raw text.
    """
    text = _MD_HEADING.sub("", text)
    text = _MD_EMPHASIS.sub("", text)
    return _WS.sub(" ", text).strip()


def parse_rewrites(text: str, k: int = MULTI_QUERY_K) -> list[str]:
    """One query per line; tolerate bullets/numbering and stray blank lines."""
    lines = [_NUMBERING.sub("", line).strip().strip('"') for line in text.splitlines()]
    return [line for line in lines if line][:k]


def cache_path(dataset: str, kind: str, instruction_id: str, model: str, collection: str = "beir") -> Path:
    """cache/<collection>/<dataset>/<kind>/<instruction>/<model>.jsonl (kind: hyde | multi_query)."""
    slug = model.replace("/", "__")
    return paths.cache_dir() / collection / dataset / kind / instruction_id / f"{slug}.jsonl"
