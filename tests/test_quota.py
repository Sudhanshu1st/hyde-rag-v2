import pytest

from hyde_rag.llm.quota import (
    Limits,
    QuotaGuardError,
    assert_free_tier,
    check_run_size,
    estimate_run,
    load_limits,
)

FREE = Limits(rpm=30, rpd=1000, tpm=8000, tpd=200_000)


def test_limits_load_for_every_role_model():
    limits = load_limits()
    assert {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"} <= set(limits)


def test_estimate_uses_the_binding_daily_limit():
    # 2,400 calls x 460 tokens = 1.104M tokens -> 5.52 days (tokens bind, not requests).
    est = estimate_run(FREE, "m", 2400, 460)
    assert est.tokens == 1_104_000
    assert est.days == pytest.approx(5.52)
    assert est.calendar_days == 6
    # Tiny calls: the 1,000 requests/day limit binds instead.
    assert estimate_run(FREE, "m", 3000, 10).days == pytest.approx(3.0)


def test_run_size_guard():
    big = estimate_run(FREE, "m", 2400, 460)
    with pytest.raises(QuotaGuardError):
        check_run_size(big, max_days=1)
    check_run_size(big, max_days=1, approved=True)
    check_run_size(estimate_run(FREE, "m", 100, 460), max_days=1)


def test_free_tier_guard(monkeypatch):
    monkeypatch.delenv("HYDE_RAG_ALLOW_PAID", raising=False)
    assert_free_tier({"x-ratelimit-limit-tokens": "8000"})
    assert_free_tier({})  # e.g. local Ollama: no limit headers, nothing billed
    with pytest.raises(QuotaGuardError):
        assert_free_tier({"X-RateLimit-Limit-Tokens": "250000"})
    monkeypatch.setenv("HYDE_RAG_ALLOW_PAID", "1")
    assert_free_tier({"x-ratelimit-limit-tokens": "250000"})
