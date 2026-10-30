"""Measure gateway overhead and availability under upstream failures.

Starts the gateway in-process with mock upstreams and fires concurrent requests
over real HTTP. Two runs:
  overhead - zero-latency upstream, so the latency measured is the gateway's own
  chaos    - the primary fails 30% of the time; retries and fallback should hide it

    python scripts/loadtest.py --requests 3000 --concurrency 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import subprocess
import sys
import tempfile
import time

import aiohttp
import httpx


def config(failure_rate: float) -> dict:
    return {
        "providers": {
            "primary": {"type": "mock", "failure_rate": failure_rate},
            "backup": {"type": "mock"},
        },
        "routes": {"m": {"primary": [{"provider": "primary", "model": "p"}],
                         "fallbacks": [{"provider": "backup", "model": "b"}]}},
        "retry": {"attempts": 2, "base_delay_s": 0.005, "max_delay_s": 0.02},
        # a 30% failure rate should not trip the breaker in this demo
        "breaker": {"failure_threshold": 1000, "cooldown_s": 1},
        "cache": {"enabled": False},
    }


def serve(cfg: dict) -> tuple[str, subprocess.Popen]:
    """Run the gateway in its own process so the load generator does not share its GIL."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    path = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    json.dump(cfg, path)  # JSON is valid YAML
    path.close()
    proc = subprocess.Popen([sys.executable, "-m", "gateway", "--config", path.name, "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(url + "/health").status_code == 200:
                return url, proc
        except httpx.HTTPError:
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError("gateway did not start")


async def run(url: str, n: int, concurrency: int) -> tuple[list[float], dict[str, int], float]:
    # aiohttp rather than httpx here: httpx's async pool slows down badly at high
    # concurrency, which would make the client, not the gateway, the bottleneck.
    lat: list[float] = []
    outcomes: dict[str, int] = {}
    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(url, connector=conn) as session:
        async def worker(ids: range) -> None:
            for i in ids:
                t0 = time.perf_counter()
                async with session.post("/v1/chat/completions", json={
                        "model": "m", "messages": [{"role": "user", "content": f"request {i}"}]}) as r:
                    await r.read()
                    lat.append(time.perf_counter() - t0)
                    key = str(r.status) if r.status != 200 else "200 via " + r.headers["x-gateway-provider"]
                    outcomes[key] = outcomes.get(key, 0) + 1

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(range(w, n, concurrency)) for w in range(concurrency)))
        return lat, outcomes, time.perf_counter() - t0


def report(name: str, lat: list[float], outcomes: dict[str, int], wall: float) -> None:
    q = statistics.quantiles(lat, n=100)
    ok = sum(v for k, v in outcomes.items() if k.startswith("200"))
    print(f"{name}: {len(lat)} requests in {wall:.2f}s ({len(lat) / wall:.0f} req/s), "
          f"p50 {q[49] * 1000:.1f} ms, p99 {q[98] * 1000:.1f} ms, success {100 * ok / len(lat):.2f}%")
    print("   outcomes:", dict(sorted(outcomes.items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=50)
    args = ap.parse_args()
    for name, rate in [("overhead", 0.0), ("chaos (primary fails 30%)", 0.3)]:
        url, proc = serve(config(rate))
        try:
            report(name, *asyncio.run(run(url, args.requests, args.concurrency)))
        finally:
            proc.terminate()
            proc.wait(10)


if __name__ == "__main__":
    main()
