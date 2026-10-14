import pytest

from hyde_rag.llm.cost import CostLimitExceeded, check_budget, estimate_cost, load_pricing


@pytest.fixture(scope="module")
def pricing():
    return load_pricing()


def test_batch_halves_list_price(pricing):
    online = estimate_cost(pricing, "claude-opus-5-5", 1000, 1000, 1000, batch=False)
    batch = estimate_cost(pricing, "claude-opus-5-5", 1000, 1000, 1000, batch=True)
    # 1M input at $4 + 1M output at $20 = $24 online, $12 in batch.
    assert online.usd == pytest.approx(24.0)
    assert batch.usd == pytest.approx(12.0)


def test_unknown_model_is_priced_as_open_weights(pricing):
    est = estimate_cost(pricing, "openai/gpt-oss-120b", 500, 100, 300)
    assert est.usd == 0.0
    assert est.input_tokens == 50_000
    assert est.output_tokens == 150_000


def test_guard_blocks_over_limit_unless_approved():
    with pytest.raises(CostLimitExceeded):
        check_budget(5.01, limit=5.0)
    check_budget(5.01, limit=5.0, approved=True)
    check_budget(4.99, limit=5.0)


def test_guard_reads_env_limit(monkeypatch):
    monkeypatch.setenv("HYDE_RAG_COST_LIMIT_USD", "1")
    with pytest.raises(CostLimitExceeded):
        check_budget(1.5)
