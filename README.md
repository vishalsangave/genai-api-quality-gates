# DriftGate — Test Suite Execution Guide

DriftGate is an API quality-gate framework: it validates HTTP APIs against their OpenAPI contract, records/replays real traffic, streams SSE telemetry, and runs property-based fuzz checks. This guide covers only how to **run the test suites**. No technical internals — that detail lives with the maintainers.

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| [uv](https://docs.astral.sh/uv/) | latest |
| Node.js + npm | ≥ 20 (for the local mock server; auto-installed by tests) |

## Step 1 — Install

```bash
git clone <this repo>
cd genai-api-quality-gates
uv sync --extra dev
```

## Step 2 — Run all tests

Nothing to start manually: the mock servers are launched automatically for the test run and stopped afterwards.

```bash
uv run pytest -m "not nightly" -v
```

Expected: **82 tests pass** in ~1.5 minutes (the stateful API tests wait out real server timers — 5 s/15 s order transitions, a 60 s export job — so most of that is by design).

Tests live in two places:

- `tests/` — the framework's own unit tests (no server needed, finishes in seconds).
- `examples/tests/` — the example service suites (order & export API, fuzz, contract, demo pipeline). These start the mock servers automatically.

## Optional — run the tests against a server you started yourself

If you want the suites to exercise **your own running instance** of the reference order & export API (for example, the original server downloaded from its public gist):

**Step A — start the reference server (from anywhere):**

```bash
git clone https://gist.github.com/sharanya-lb/b429b6e807f95a8df8216c4343ff6766 gist && cd gist
npm install express && node server.js        # serves http://localhost:3000/v1
```

**Step B — from the DriftGate repo, point the tests at it:**

```bash
DRIFT_GATE_EXPRESS_URL=http://localhost:3000/v1 uv run pytest -m "not nightly" -v
```

Without that variable the suites start their own copy automatically; with it, they run against yours and never touch its lifecycle. If the variable is set but nothing is listening, the run fails immediately with instructions instead of hanging.

## Useful run variations

```bash
# Fast subset only (skips the 5s/15s/60s stateful lifecycle tests):
uv run pytest -m "not nightly and not slow" -v

# One suite:
uv run pytest examples/tests/test_order_export_service.py -v

# The stateful lifecycle tests by themselves (~80s):
uv run pytest examples/tests/test_order_export_service.py -m slow -v

# Property-based fuzz suite (generated adversarial inputs):
uv run pytest examples/tests/test_order_export_service_fuzz.py -v

# Config-driven contract validation:
uv run pytest examples/tests/test_config_driven_order_export.py \
  --driftgate-config examples/driftgate.service.template.yaml -v

# Framework CLI: run tests, then the release-gate decision
# (exit codes: PROMOTE=0, ROLLBACK=1, HOLD=2):
uv run driftgate run --config driftgate.config.yaml

uv run driftgate version
```

## What the suites cover

- **Order & export API suites** — authentication, order creation and totals, the 5s/15s order state machine, cancellation rules (409 after completion), the 60-second export job, and CSV download content.
- **Fuzz suite** — machine-generated bad inputs against the live API; the API must always answer with a declared error, never crash.
- **Contract suites** — every response validated against the API's OpenAPI spec.
- **Framework suites** — the quality-gate machinery itself (contracts, replay, streaming, statistics).

## CI

Every pull request runs the full suite automatically (zero API keys, no external calls). Results are stored as workflow artifacts for 90 days.

## More documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — maintainer-facing deep dive: the three-layer design, statistical release governor, configuration reference, and roadmap.
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — contributing guide: repo layout, ground rules, how to add tests and relations, quality gates.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `npm install` errors on first run | Ensure Node.js ≥ 18 is installed; the mock's deps install automatically on first run |
| Tests hang waiting for a server | Unset `DRIFT_GATE_EXPRESS_URL` to use auto-start, or start your server and re-check the URL |
| `RuntimeError: DRIFT_GATE_EXPRESS_URL ... no order/export API is answering` | Start the server (`npm install express && node server.js`) or unset the variable |
| Port already in use | The auto-started mock picks a free port; kill any stray `node server.js` you started manually |

## License

MIT — see [LICENSE](LICENSE).
