"""Fixtures for the example service test suites.

Service tests live under ``examples/tests/`` — separate from the framework's
own unit tests in ``tests/`` — so a user running ``uv run pytest`` against
their installation knows which suites validate DriftGate itself and which
validate the vendored example APIs.

Fixtures here:

- ``express_base_url`` — starts (or reuses) the stateful order & export API.
- ``express_client`` — httpx client with the API's contract validators.
- ``demo_base_url`` — in-process uvicorn server for the GenAI demo service.
- ``poll_until`` — the shared long-running-poll helper.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import httpx
import pytest
import uvicorn
import yaml

from driftgate.contracts.validator import ContractValidator
from examples.demo_service.app import app as demo_app

ROOT = Path(__file__).resolve().parents[2]
EXPRESS_DIR = ROOT / "examples" / "express_mock"
TEMPLATE = ROOT / "examples" / "driftgate.service.template.yaml"
T = TypeVar("T")


class ExpressClient(httpx.Client):
    """httpx client with the suite's precompiled contract validators attached."""

    contracts: dict[str, ContractValidator]


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
    raise TimeoutError(
        f"Condition did not become true within {timeout_s:.1f}s; last value: {last!r}"
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


EXTERNAL_EXPRESS_URL_ENV = "DRIFT_GATE_EXPRESS_URL"


def _probe_v1(base_url: str) -> bool:
    """True when an order/export server answers a protected route as expected.

    Works against the original gist too: an unauthenticated POST /v1/orders
    must return 401 with a JSON error body — a signature no other server is
    likely to share, so /healthz (our vendored addition) is never required.
    """
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.post(f"{base_url}/orders", json={})
            return response.status_code == 401
    except httpx.TransportError:
        return False


@pytest.fixture(scope="session")
def express_base_url() -> str:
    """Run the stateful order/export mock for live API tests.

    Two modes:

    - **External server**: set ``DRIFT_GATE_EXPRESS_URL`` (e.g.
      ``http://localhost:3000/v1`` after running the original gist's
      ``npm install express && node server.js``). The fixture verifies the
      API is live and answering with its signature, then tests run against
      your server, which is never stopped.
    - **Auto-spawned** (default): installs ``npm`` deps if needed and starts
      the vendored behavior-identical copy on a free port, stopping it at
      session end.
    """
    external = os.environ.get(EXTERNAL_EXPRESS_URL_ENV)
    if external:
        base = external.rstrip("/")
        if not _probe_v1(base):
            raise RuntimeError(
                f"{EXTERNAL_EXPRESS_URL_ENV}={external!r} was set, but no order/export API is "
                f"answering there (POST /orders did not return 401 as expected). Start the "
                f"server (npm install express && node server.js) or unset the variable."
            )
        yield base
        return

    if not (EXPRESS_DIR / "node_modules" / "express").exists():
        subprocess.run(
            ["npm", "install"], cwd=EXPRESS_DIR, check=True, capture_output=True, text=True
        )
    port = _free_port()
    env = {**os.environ, "PORT": str(port)}
    # A pipe that no one drains blocks a chatty child; route output to DEVNULL
    # so a large npm banner or error trace can never wedge readiness polling.
    process = subprocess.Popen(
        ["node", "server.js"],
        cwd=EXPRESS_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}/v1"

    def ready() -> bool:
        try:
            with httpx.Client(timeout=1.0) as client:
                return client.get(f"http://127.0.0.1:{port}/healthz").status_code == 200
        except httpx.TransportError:
            return False

    try:
        poll_until(ready, lambda ok: ok, interval_s=0.1, timeout_s=10.0)
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(scope="session")
def express_contract() -> dict[str, ContractValidator]:
    with (EXPRESS_DIR / "openapi.yaml").open() as handle:
        spec = yaml.safe_load(handle)
    login = ContractValidator(spec)
    order_create = ContractValidator(spec)
    return {"login": login, "order_create": order_create}


@pytest.fixture(scope="session")
def express_client(
    express_base_url: str, express_contract: dict[str, ContractValidator]
) -> ExpressClient:
    client = ExpressClient(base_url=express_base_url, timeout=10.0)
    client.contracts = express_contract  # type: ignore[attr-defined]
    yield client
    client.close()


@pytest.fixture(scope="session")
def demo_base_url() -> str:
    """Serve the GenAI demo app in-process over an ephemeral port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(demo_app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=5.0) as client:
        while True:
            try:
                if client.get(f"{base}/healthz").status_code == 200:
                    break
            except httpx.TransportError:
                pass
    yield base
    server.should_exit = True
    thread.join(timeout=5)
