"""Check that your API key works on the FREE tier and that every configured model is available.

Uses ~1 tiny request per model (a few hundred tokens in total). Prints the real rate limits
from the response headers. Never prints the key.

Usage:
    uv run python scripts/check_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hyde_rag.llm.client import LLMError, LLMRequest, OpenAICompatClient  # noqa: E402
from hyde_rag.llm.quota import QuotaGuardError  # noqa: E402


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "llm.yaml").read_text(encoding="utf-8"))
    models = sorted(set(cfg["roles"].values()))
    client = OpenAICompatClient()
    print(f"Endpoint: {client.base_url}")

    available = {m.id for m in client._client.models.list()}
    missing = [m for m in models if m not in available]
    print(f"Models visible to this key: {len(available)}; configured: {', '.join(models)}")
    if missing:
        print(f"NOT AVAILABLE: {', '.join(missing)}")

    ok = True
    for model in models:
        if model in missing:
            ok = False
            continue
        reasoning = cfg.get("generation", {}).get(model, {}).get("reasoning_effort")
        req = LLMRequest(
            model=model,
            messages=(("user", "Reply with the single word: ready"),),
            max_tokens=300,
            reasoning_effort=reasoning,
        )
        try:
            r = client.complete(req)
        except QuotaGuardError as e:
            print(f"STOP: {e}")
            return 2
        except LLMError as e:
            print(f"{model}: FAILED: {e}")
            ok = False
            continue
        h = r.headers
        print(
            f"{model}: '{r.text[:20]}' in {r.latency_s:.2f}s, "
            f"{r.prompt_tokens}+{r.completion_tokens} tokens | "
            f"limits: {h.get('x-ratelimit-limit-requests', '?')} req/day, "
            f"{h.get('x-ratelimit-limit-tokens', '?')} tokens/min | "
            f"remaining today: {h.get('x-ratelimit-remaining-requests', '?')} req"
        )
    print("Free-tier check passed." if ok else "Some models failed; see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
