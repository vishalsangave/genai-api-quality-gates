from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPRESS_DIR = ROOT / "examples" / "express_mock"
T = TypeVar("T")


def poll_until(
    fetch: Callable[[], T],
    predicate: Callable[[T], bool],
    *,
    interval_s: float,
    timeout_s: float,
) -> T:
    """Poll a stateful endpoint and retain the last response in timeout errors."""
    deadline = time.monotonic() + timeout_s
    last: T | None = None
    while time.monotonic() < deadline:
        last = fetch()
        if predicate(last):
            return last
        time.sleep(interval_s)
    raise TimeoutError(f"Condition did not become true within {timeout_s:.1f}s; last value: {last!r}")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def express_base_url() -> str:
    """Run the vendored stateful order/export mock for live API tests."""
    if not (EXPRESS_DIR / "node_modules" / "express").exists():
        subprocess.run(["npm", "install"], cwd=EXPRESS_DIR, check=True, capture_output=True, text=True)
    port = _free_port()
    env = {**os.environ, "PORT": str(port)}
    process = subprocess.Popen(
        ["node", "server.js"],
        cwd=EXPRESS_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}/v1"
    try:
        with httpx.Client(timeout=1.0) as client:
            poll_until(
                lambda: client.get(f"http://127.0.0.1:{port}/healthz"),
                lambda response: response.status_code == 200,
                interval_s=0.1,
                timeout_s=10.0,
            )
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
