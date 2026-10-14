"""Token-cost estimation and the bulk-run cost guard.

Every bulk LLM job (HyDE generation, multi-query, answers, judging, synthetic
queries) must call :func:`check_budget` before submitting work. If the
estimate exceeds the limit (default $5, env ``HYDE_RAG_COST_LIMIT_USD``) the
run stops with :class:`CostLimitExceeded` unless it was explicitly approved.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[3] / "configs" / "pricing.yaml"
DEFAULT_LIMIT_USD = 5.0


class CostLimitExceeded(RuntimeError):
    """Raised when a bulk run's estimated cost is above the approved limit."""


@dataclass(frozen=True)
class ModelPrice:
    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens (thinking tokens bill as output)
    cache_read: float = 0.0


@dataclass(frozen=True)
class Pricing:
    models: dict[str, ModelPrice]
    batch_discount: float
    source: str
    verified: str

    def price_for(self, model: str) -> ModelPrice:
        if model in self.models:
            return self.models[model]
        # Anything served through the OpenAI-compatible backend is open-weights.
        return self.models["open-weights"]


def load_pricing(path: Path = DEFAULT_PRICING_PATH) -> Pricing:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models = {name: ModelPrice(**vals) for name, vals in raw["models"].items()}
    return Pricing(
        models=models,
        batch_discount=float(raw["batch_discount"]),
        source=raw["source"],
        verified=str(raw["verified"]),
    )


@dataclass(frozen=True)
class CostEstimate:
    model: str
    n_calls: int
    input_tokens: int
    output_tokens: int
    batch: bool
    usd: float


def estimate_cost(
    pricing: Pricing,
    model: str,
    n_calls: int,
    input_tokens_per_call: float,
    output_tokens_per_call: float,
    *,
    batch: bool = True,
) -> CostEstimate:
    """Estimate USD for ``n_calls`` requests with the given mean token counts."""
    price = pricing.price_for(model)
    total_in = round(n_calls * input_tokens_per_call)
    total_out = round(n_calls * output_tokens_per_call)
    usd = (total_in * price.input + total_out * price.output) / 1_000_000
    if batch:
        usd *= pricing.batch_discount
    return CostEstimate(model, n_calls, total_in, total_out, batch, usd)


def cost_limit_usd() -> float:
    return float(os.environ.get("HYDE_RAG_COST_LIMIT_USD", DEFAULT_LIMIT_USD))


def check_budget(estimate_usd: float, *, approved: bool = False, limit: float | None = None) -> None:
    """Stop a bulk run whose estimate is above the limit unless explicitly approved."""
    limit = cost_limit_usd() if limit is None else limit
    if estimate_usd > limit and not approved:
        raise CostLimitExceeded(
            f"Estimated cost ${estimate_usd:.2f} exceeds the ${limit:.2f} limit. "
            "Get approval, then re-run with --approve-cost."
        )
