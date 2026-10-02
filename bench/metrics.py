"""Scrape vLLM's Prometheus /metrics endpoint into server_metrics.jsonl rows."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

import httpx

# Histogram buckets and creation timestamps are bulky and not used in analysis.
_SKIP_SUFFIXES = ("_bucket", "_created")


def parse_prometheus(text: str, prefix: str = "vllm:") -> dict[str, float]:
    """Return {sample name with labels: value} for samples whose name starts with prefix."""
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        key, _, value = line.rpartition(" ")
        name = key.split("{", 1)[0]
        if name.endswith(_SKIP_SUFFIXES):
            continue
        try:
            samples[key] = float(value)
        except ValueError:
            continue
    return samples


def metric(samples: dict[str, float], name: str) -> float | None:
    """Sum a metric across label sets (single engine => usually exactly one)."""
    values = [v for k, v in samples.items() if k == name or k.startswith(name + "{")]
    return sum(values) if values else None


class MetricsScraper:
    """Periodic background scrape plus on-demand scrapes at turn boundaries."""

    def __init__(
        self, metrics_url: str, interval_s: float, sink: Callable[[dict[str, Any]], None]
    ) -> None:
        self.metrics_url = metrics_url
        self.interval_s = interval_s
        self.sink = sink
        self.context: dict[str, Any] = {}
        self._client = httpx.AsyncClient(timeout=interval_s)
        self._task: asyncio.Task[None] | None = None

    async def scrape(self, kind: str, **extra: Any) -> dict[str, float] | None:
        try:
            response = await self._client.get(self.metrics_url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            self.sink({"t": time.time(), "kind": kind, **self.context, **extra, "error": str(e)})
            return None
        samples = parse_prometheus(response.text)
        self.sink({"t": time.time(), "kind": kind, **self.context, **extra, "metrics": samples})
        return samples

    async def _loop(self) -> None:
        while True:
            await self.scrape("periodic")
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._client.aclose()
