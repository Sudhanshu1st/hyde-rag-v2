"""Free-tier bulk runner: pace, pilot, guard, cache, resume (D-LLM-2, D-LLM-4).

Work items already in the cache are skipped, so a run can be stopped at any time and
re-started later (including the next day, when free quota resets) without repeating calls.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hyde_rag.llm.cache import JsonlCache, cache_key
from hyde_rag.llm.client import (
    LLMClient,
    LLMRequest,
    PermanentLLMError,
    RateLimited,
    TransientLLMError,
)
from hyde_rag.llm.quota import Limits, check_run_size, estimate_run

PILOT_CALLS = 20
MAX_TRANSIENT_RETRIES = 5
DEFAULT_RETRY_AFTER_S = 60.0


@dataclass(frozen=True)
class WorkItem:
    item_id: str  # e.g. a query id
    sample: int  # sample index (HyDE draws N independent samples)
    request: LLMRequest


@dataclass
class RunReport:
    total: int = 0
    already_cached: int = 0
    completed: int = 0
    failed: int = 0
    tokens: int = 0

    def __str__(self) -> str:
        return (
            f"{self.completed} new, {self.already_cached} cached, {self.failed} failed "
            f"of {self.total}; {self.tokens:,} tokens used"
        )


class Pacer:
    """Client-side sliding-window limiter for requests/min and tokens/min."""

    def __init__(
        self,
        limits: Limits,
        *,
        headroom: float = 0.9,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm = max(1, int(limits.rpm * headroom))
        self.tpm = max(1, int(limits.tpm * headroom))
        self._events: deque[tuple[float, int]] = deque()
        self._clock, self._sleep = clock, sleep

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def wait(self, expected_tokens: int) -> None:
        while True:
            now = self._clock()
            self._trim(now)
            used = sum(t for _, t in self._events)
            if len(self._events) < self.rpm and used + expected_tokens <= self.tpm:
                return
            if not self._events:
                return  # a single call larger than the per-minute budget: let the API decide
            self._sleep(max(0.05, 60.0 - (now - self._events[0][0])))

    def record(self, tokens: int) -> None:
        self._events.append((self._clock(), tokens))


def run_bulk(
    client: LLMClient,
    cache: JsonlCache,
    items: Sequence[WorkItem],
    limits: Limits,
    *,
    approved: bool = False,
    max_days: float | None = None,
    pilot: int = PILOT_CALLS,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> RunReport:
    report = RunReport(total=len(items))
    keyed = [(cache_key(client.base_url, it.request, it.sample), it) for it in items]
    todo = [(k, it) for k, it in keyed if k not in cache]
    report.already_cached = len(items) - len(todo)
    if not todo:
        log(f"Nothing to do: all {len(items)} items are cached.")
        return report

    model = todo[0][1].request.model
    pacer = Pacer(limits, clock=clock, sleep=sleep)
    mean_tokens = 500.0
    measured: list[int] = []

    for n, (key, item) in enumerate(todo):
        if n == min(pilot, len(todo)) and n < len(todo):
            # Pilot done: decide with measured tokens whether the rest may run.
            per_call = sum(measured) / len(measured) if measured else mean_tokens
            est = estimate_run(limits, model, len(todo) - n, per_call)
            log(
                f"[{model}] pilot: {per_call:.0f} tokens/call -> remaining {est.requests:,} calls "
                f"= {est.tokens:,} tokens = {est.days:.1f} days of free quota"
            )
            check_run_size(est, approved=approved, max_days=max_days)

        attempts = 0
        while True:
            pacer.wait(int(mean_tokens))
            try:
                response = client.complete(item.request)
            except RateLimited as e:
                wait = e.retry_after_s if e.retry_after_s is not None else DEFAULT_RETRY_AFTER_S
                log(f"[{model}] free-tier limit reached; waiting {wait:,.0f}s (progress is saved)")
                pacer.record(0)
                sleep(wait)
                continue
            except TransientLLMError as e:
                attempts += 1
                if attempts > MAX_TRANSIENT_RETRIES:
                    log(f"[{model}] giving up on {item.item_id}#{item.sample}: {e}")
                    report.failed += 1
                    break
                sleep(min(60.0, 2.0**attempts))
                continue
            except PermanentLLMError as e:
                log(f"[{model}] request rejected for {item.item_id}#{item.sample}: {e}")
                report.failed += 1
                break
            cache.put(key, item.item_id, item.sample, response)
            pacer.record(response.total_tokens)
            measured.append(response.total_tokens)
            mean_tokens = sum(measured[-50:]) / len(measured[-50:])
            report.completed += 1
            report.tokens += response.total_tokens
            if report.completed % 50 == 0:
                log(f"[{model}] {report.completed + report.already_cached}/{report.total} done")
            break

    log(f"[{model}] {report}")
    return report
