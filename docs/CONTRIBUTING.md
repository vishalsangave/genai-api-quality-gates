# Contributing to DriftGate

Thanks for contributing! This guide covers the ground rules: where code lives, how tests are organized, the style gates your change must pass, and how documentation is split. Read [ARCHITECTURE.md](ARCHITECTURE.md) first for the design rationale — this guide assumes you've skimmed it.

## Repository layout

```
driftgate/               the installable package (no service code here)
  config.py              declarative manifest (driftgate.config.yaml) + env overlay
  cli.py                 driftgate run/record/gate/infer-schema/version
  pytest_plugin.py       pytest11 entry point: fixtures, options, metrics hook
  contracts/             Layer 1: validator.py (gate), inference.py (bootstrap)
  recorder/              Layer 2: fingerprint.py, cassette.py, interceptor.py
  streaming/probe.py     SSE frame parser + TTFT/ITL telemetry
  metamorphic/           Layer 3: relations.py (mutators), evaluator.py (embedder/judges)
  governor/              statistics.py (bootstrap CI, Mann-Kendall), gate.py (tri-state)
examples/
  demo_service/          GenAI demo (SSE + fault injection)
  express_mock/          vendored stateful order & export API
  driftgate.service.template.yaml   fully-documented manifest (under test!)
docs/                    ARCHITECTURE.md (this doc's deep-dive sibling), guides
tests/                   framework unit tests only — no server, no fixtures
examples/tests/          service suites (order/export, fuzz, inference, pipeline)
```

**The split that matters:** anything validating DriftGate itself goes in `tests/`; anything exercising a service goes in `examples/tests/` with its fixtures in `examples/tests/conftest.py`. Don't mix them — `uv run pytest tests/` must stay server-free and finish in under a second.

## Setup

```bash
git clone <repo> && cd genai-api-quality-gates
uv sync --extra dev      # creates .venv with runtime + dev deps
uv run pytest -q         # sanity: 84 tests pass (framework in <1s, service suites ~80s)
```

## The ground rules

These come from design decisions recorded in ARCHITECTURE.md. A change that violates one will be rejected in review — check here first:

1. **PR path stays deterministic and offline.** No network calls, no SDK imports, no LLM calls in anything reachable by `-m "not nightly"`. Layer 3's real judge is an injected transport callable — the package never imports `openai`/`anthropic`.
2. **Everything seeded is reproducible.** Any randomness takes a seed (`rng_seed: 20260917` default). CI runs must produce byte-identical decisions across machines.
3. **Fingerprinting never throws.** `fingerprint_request` must return a digest for any input, however malformed. Parse failures fall back (AST → whitespace collapse), they don't raise.
4. **Judges fail closed.** Unparsable judge output = FAIL, never PASS. Binary grading only.
5. **Secrets redacted before persistence.** Anything touching cassette storage goes through `redact_headers()`. Never persist `authorization`/cookies/api keys verbatim.
6. **Cassettes for stateless routes only.** Time-driven server state (the order state machine, export jobs) always runs live — replaying it would lie.
7. **The SSE parser is spec-correct on line terminators**: `\n`, `\r\n`, and bare `\r` are all line breaks; a chunk-boundary `\r` must stay pending. Regression tests pin this — extend, don't weaken.
8. **The config template is under test.** If you rename any key in `driftgate/config.py`, `examples/driftgate.service.template.yaml` and `tests/../examples/tests/test_config_driven_order_export.py` must be updated in the same change — the rot-guard suite will fail otherwise.
9. **CLI never owns service lifecycle.** `driftgate` commands require `service.base_url` to already be reachable; spawning servers is the caller's job (test fixtures, CI steps).
10. **The build-report pipeline never breaks a run.** Every hook the report uses (`pytest_runtest_logstart/logreport/logfinish`, the `httpx.Client.send` wrapper, `pytest_sessionfinish`'s report write) must swallow its own exceptions into a stderr warning. A reporting bug changes nothing about pytest's exit code. The wrapper is strict pass-through: it never mutates requests, never raises into the caller, and restores the original `send` by identity at `pytest_unconfigure`.
11. **Low-comment discipline.** Rationale, design trade-offs, and known-defect explanations belong in `docs/ARCHITECTURE.md`, not in code. A one-line comment pointing at the doc is the maximum; module docstrings state what a module *is*, not why it exists.

## Adding code

### A new feature in the package

1. Add the module (or extend an existing one) with full type hints, Pydantic v2 for models, `StrEnum` for enums.
2. Add unit tests in `tests/test_<module>.py`. Framework tests must not start servers.
3. If the feature adds config keys, update `examples/driftgate.service.template.yaml` + the rot-guard test.
4. Update `docs/ARCHITECTURE.md` with the design rationale (why, not how).

### Adding to the build report

The report (`.driftgate/report.json`/`report.md`, built by `driftgate/reporting.py` from the plugin's sessionfinish pipeline) is the QA-facing artifact. To surface a new signal in it:

1. Compute the value as a frozen dataclass in `reporting.py` with a `to_dict`/`from_dict` round-trip on `ReportData` — the CLI's `driftgate report` re-renders *from stored JSON*, so the round-trip must be lossless (there is a test pinning this).
2. Extend `render_markdown` for the human-readable section; keep the output deterministic (no timestamps beyond the header, no dict-iteration order).
3. Cap everything: any per-test collection must stay bounded like `CallTracker` (see `PER_TEST_CALL_CAP`/`MAX_TRACKED_TESTS`) so fuzz suites can't blow memory.
4. The spec-diff column of the report comes from `driftgate/specdiff.py`; if you add a finding kind there, extend the severity table and its tests, and document the classification in ARCHITECTURE.md's "Build report" section.

### Adding a metamorphic relation

1. Implement in `driftgate/metamorphic/relations.py` — deterministic, seeded via the passed `rng`, returns `[]` when no change applies (never a silently-identical variant).
2. Register the name in `_SUPPORTED_RELATIONS` (`driftgate/config.py`) and in `build_relations`.
3. Decide the oracle type (cosine invariance / rubric judgment / structural validity) and document it in ARCHITECTURE.md's Layer 3 section.
4. Add relations tests: seeded determinism, identity-fallback behavior, and dispatch to the right oracle.

### Adding an example service suite

Service suites live in `examples/tests/`. Conventions:

- Fixtures come from `examples/tests/conftest.py` (`express_base_url`, `express_client`, `demo_base_url`, `poll_until`). Extend conftest rather than redefining fixtures per file.
- Poll long-running state with `poll_until`, never `time.sleep` loops with magic durations.
- ID threading: take IDs from responses, never hardcode (`ORD-10001`-style IDs only for the literal CSV-body assertions).
- Mark real-time tests `@pytest.mark.slow`.
- The safety oracle for fuzz tests: only declared statuses, schema-valid bodies, never 5xx — drive `ContractValidator` from the service's spec, don't hand-roll assertions.

## Running the quality gates locally

```bash
uv run pytest -q                     # everything (testpaths covers both dirs)
uv run pytest tests/ -q              # framework only (<1s — run this constantly)
uv run pytest examples/tests/ -q     # service suites (~80s)
uv run ruff check tests examples driftgate
uv run ruff format --check tests examples driftgate
```

CI (`.github/workflows/driftgate-gate.yml`) runs the same commands: `pr-gate` on every PR (structural, $0), `nightly-eval` on cron/label. Your PR must pass ruff + the full suite exactly as CI runs them.

## Documentation split

| Document | Audience | Contains |
|---|---|---|
| `README.md` (root) | users | setup + execution commands only |
| `docs/ARCHITECTURE.md` | maintainers | design rationale, layer mechanics, governor rules, inference guarantees, known defects, roadmap |
| `docs/CONTRIBUTING.md` | contributors | this guide |

When your change alters behavior described in ARCHITECTURE.md, update that file in the same PR. The known-defects section is load-bearing: if you fix the example service's NaN `totalAmount` defect, the pin test and that section must both change.

## Commit style

Imperative one-line subject (`fingerprint: tolerate bare CR as line terminator`), no issue-number noise. Commits should be one logical change; test + implementation land together.

## Reporting bugs

Include: the exact command run, the failing test(s), and — for framework bugs — a minimal reproduction. For defects in the example service, check ARCHITECTURE.md's known-defects section first: the NaN defect is documented and pinned intentionally.
