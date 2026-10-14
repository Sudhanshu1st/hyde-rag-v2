"""Free-tier quota planning and the zero-spend guard.

This project never spends money. Instead of a dollar budget, each bulk LLM run is
checked against the free plan's per-model daily limits:

* :func:`estimate_run` turns a run into requests, tokens and days of free quota.
* :func:`check_run_size` stops runs longer than ``max_days`` of quota unless approved.
* :func:`assert_free_tier` refuses to continue if the API key's rate-limit headers show
  paid-plan limits, so a key with billing enabled can't run up charges by accident.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from hyde_rag import paths

DEFAULT_LLM_CONFIG = paths.CONFIG_DIR / "llm.yaml"

# Groq's free plan allows 8K tokens/minute per model; its paid Developer plan starts at
# 250K. Anything above this threshold means the key is not on the free plan.
FREE_TIER_MAX_TPM = 50_000


class QuotaGuardError(RuntimeError):
    """Raised when a run is too large to start without approval, or the key isn't free-tier."""


@dataclass(frozen=True)
class Limits:
    rpm: int
    rpd: int
    tpm: int
    tpd: int


def load_limits(path: Path = DEFAULT_LLM_CONFIG) -> dict[str, Limits]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {model: Limits(**vals) for model, vals in raw["free_tier_limits"].items()}


@dataclass(frozen=True)
class RunEstimate:
    model: str
    requests: int
    tokens: int
    days: float  # days of free quota for this model (the binding daily limit)
    active_minutes: float  # minimum wall-clock minutes if the per-minute limits bind

    @property
    def calendar_days(self) -> int:
        return max(1, math.ceil(self.days))


def estimate_run(limits: Limits, model: str, requests: int, tokens_per_call: float) -> RunEstimate:
    tokens = round(requests * tokens_per_call)
    days = max(tokens / limits.tpd, requests / limits.rpd)
    minutes = max(tokens / limits.tpm, requests / limits.rpm)
    return RunEstimate(model, requests, tokens, days, minutes)


def check_run_size(estimate: RunEstimate, *, approved: bool = False, max_days: float | None = None) -> None:
    """Stop runs that would occupy more than ``max_days`` of a model's free quota."""
    if max_days is None:
        max_days = float(os.environ.get("HYDE_RAG_MAX_QUOTA_DAYS", "1"))
    if estimate.days > max_days and not approved:
        raise QuotaGuardError(
            f"{estimate.model}: {estimate.requests:,} requests / {estimate.tokens:,} tokens is "
            f"{estimate.days:.1f} days of free quota (limit {max_days:g}). "
            "It will resume across days automatically; re-run with --yes to start it."
        )


def assert_free_tier(headers: dict[str, str]) -> None:
    """Refuse to continue if rate-limit headers show a paid plan.

    Groq returns ``x-ratelimit-limit-tokens`` (tokens per minute) on every response.
    Set HYDE_RAG_ALLOW_PAID=1 only if you deliberately want to use a paid plan.
    """
    if os.environ.get("HYDE_RAG_ALLOW_PAID") == "1":
        return
    lowered = {k.lower(): v for k, v in headers.items()}
    raw = lowered.get("x-ratelimit-limit-tokens")
    if raw is None:
        return  # backend doesn't report limits (e.g. local Ollama): nothing is billed
    try:
        tpm = int(float(raw))
    except ValueError:
        return
    if tpm > FREE_TIER_MAX_TPM:
        raise QuotaGuardError(
            f"This API key reports {tpm:,} tokens/minute, which is a paid-plan limit. "
            "This project only runs on the free tier; use a free-plan key "
            "(or set HYDE_RAG_ALLOW_PAID=1 if you really mean to pay)."
        )
