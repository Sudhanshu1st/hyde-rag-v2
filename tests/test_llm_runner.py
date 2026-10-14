import pytest

from hyde_rag.llm.cache import JsonlCache, cache_key
from hyde_rag.llm.client import LLMRequest, LLMResponse, PermanentLLMError, RateLimited
from hyde_rag.llm.quota import Limits, QuotaGuardError
from hyde_rag.llm.runner import Pacer, WorkItem, run_bulk

FREE = Limits(rpm=30, rpd=1000, tpm=8000, tpd=200_000)


class FakeClient:
    """Deterministic mock LLM: echoes the item, optionally failing on scripted calls."""

    base_url = "https://fake.local/v1"

    def __init__(self, tokens=400, script=None):
        self.calls = 0
        self.tokens = tokens
        self.script = list(script or [])

    def complete(self, request):
        self.calls += 1
        if self.script:
            action = self.script.pop(0)
            if action is not None:
                raise action
        prompt = request.messages[-1][1]
        return LLMResponse(f"passage for {prompt}", "stop", request.model, 100, self.tokens - 100, 0.01)


class FakeTime:
    """Virtual clock: sleeping advances time instantly, so pacing logic runs for real."""

    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def items(n, samples=1):
    return [
        WorkItem(f"q{i}", s, LLMRequest("m", (("user", f"query {i}"),), seed=s))
        for i in range(n)
        for s in range(samples)
    ]


def run(client, cache, work, **kwargs):
    t = kwargs.pop("time", FakeTime())
    return run_bulk(client, cache, work, FREE, sleep=t.sleep, clock=t.clock, log=lambda _: None, **kwargs)


def test_run_caches_and_resumes(tmp_path):
    client = FakeClient()
    report = run(client, JsonlCache(tmp_path / "c.jsonl"), items(3, samples=2))
    assert report.completed == 6 and client.calls == 6
    # A fresh process sees the same cache on disk and makes no new calls.
    again = run(client, JsonlCache(tmp_path / "c.jsonl"), items(3, samples=2))
    assert again.already_cached == 6 and client.calls == 6


def test_cache_records_store_no_prompt_text(tmp_path):
    run(FakeClient(), JsonlCache(tmp_path / "c.jsonl"), items(1))
    raw = (tmp_path / "c.jsonl").read_text(encoding="utf-8")
    assert '"item_id": "q0"' in raw and "query 0" not in raw.replace("passage for query 0", "")


def test_sample_index_changes_the_key():
    req = LLMRequest("m", (("user", "x"),))
    assert cache_key("u", req, 0) != cache_key("u", req, 1)


def test_rate_limit_waits_then_continues(tmp_path):
    t = FakeTime()
    report = run(
        FakeClient(script=[RateLimited(30.0), None]), JsonlCache(tmp_path / "c.jsonl"), items(1), time=t
    )
    assert report.completed == 1 and 30.0 in t.slept


def test_permanent_errors_are_skipped_not_retried(tmp_path):
    client = FakeClient(script=[PermanentLLMError("400: bad model")])
    report = run(client, JsonlCache(tmp_path / "c.jsonl"), items(2))
    assert report.failed == 1 and report.completed == 1 and client.calls == 2


def test_pilot_then_quota_guard_stops_large_runs(tmp_path):
    cache = JsonlCache(tmp_path / "c.jsonl")
    # 600 calls x 400 tokens = 240K tokens > 1 day of a 200K/day model.
    with pytest.raises(QuotaGuardError):
        run(FakeClient(), cache, items(600), max_days=1)
    assert len(cache) == 20  # only the pilot ran, and it is kept
    approved = run(FakeClient(), cache, items(600), max_days=1, approved=True)
    assert approved.already_cached == 20 and approved.completed == 580


def test_bulk_run_stays_under_free_tier_tokens_per_minute(tmp_path):
    t = FakeTime()
    run(FakeClient(tokens=400), JsonlCache(tmp_path / "c.jsonl"), items(100), time=t)
    # 100 calls x 400 tokens = 40K tokens at <= 7.2K/min (90% of 8K) -> at least ~5 minutes.
    assert t.now >= 5 * 60


def test_pacer_respects_requests_per_minute():
    t = FakeTime()
    pacer = Pacer(Limits(rpm=2, rpd=10, tpm=10_000, tpd=10**6), headroom=1.0, clock=t.clock, sleep=t.sleep)
    for _ in range(3):
        pacer.wait(10)
        pacer.record(10)
    assert t.slept and t.now >= 60.0  # the third request waited for the window to roll over


def test_pacer_spacing_terminates_with_float_residue():
    # Regression: a large clock value plus a tiny remaining gap must not spin forever.
    t = FakeTime()
    t.now = 1_000_000.123
    pacer = Pacer(Limits(rpm=27, rpd=1000, tpm=10**6, tpd=10**9), headroom=1.0, clock=t.clock, sleep=t.sleep)
    for _ in range(5):
        pacer.wait(10)
        pacer.record(10)
    assert t.now - 1_000_000.123 >= 4 * 60 / 27 - 1e-6  # evenly spaced, ~2.2 s apart
