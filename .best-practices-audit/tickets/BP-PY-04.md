# BP-PY-04: Add a local coverage gate with branch coverage to pyproject.toml

> **Status: WON'T DO (permanent).** Closed without action by maintainer decision.

**Severity:** medium  |  **Rule:** PY-035 (python-best-practices)  |  **Area:** `pyproject.toml`, dev tooling / test config

## Context
The repo has a large, mature test suite — 11 test modules totalling ~5,685 lines, with `tests/test_entity.py` alone at ~2,022 lines — and `CLAUDE.md` explicitly mandates "Test all error paths: connection refused, auth failure, timeout, malformed response." Despite this, there is no coverage measurement at all: `pyproject.toml` has no `[tool.coverage.*]` sections, and `pytest-cov` is absent from the `dev` dependency group. There is also no `.coveragerc`, `setup.cfg`, or `tox.ini` carrying coverage config.

Without a configured gate, nothing flags when a new error branch ships untested. PY-035 specifically wants the gate in `pyproject.toml` (not CI-only) so it runs locally during development, and wants **branch** coverage enabled because line coverage hides happy-path bias — exactly the failure mode the CLAUDE.md error-path mandate is trying to prevent. The project is well past the "<100 lines, too noisy" exemption in the rule, so the gate is warranted.

This is a tooling/config gap, not a code defect. The fix is confined to `pyproject.toml` plus a lockfile refresh.

## Current state
`pyproject.toml` — the `dev` group has no coverage tool, and there are no coverage sections anywhere in the file.

`/workspace/LLM-Home-Controller/pyproject.toml:13-26`
```toml
[dependency-groups]
dev = [
    "hassil>=3.0",
    "home-assistant-intents>=2026.2.3",
    "litellm<=1.82.6",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-homeassistant-custom-component",
    "ruff>=0.9",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

Source code under `custom_components/llm_home_controller/` contains ellipsis-body Protocol/abstract definitions (e.g. `providers/__init__.py` lines 18, 30, 45, 49, 53, 60, 69) that should be excluded from coverage. No `if TYPE_CHECKING:` blocks exist today, but they are idiomatic in Home Assistant code and likely to appear, so the exclusion is included pre-emptively per the rule's recommended list.

This is a uv-managed project (`[dependency-groups]` table + `uv.lock` present), so the new dev dep must be added via uv and the lockfile regenerated.

## Proposed fix
Add `pytest-cov` to the `dev` group and two coverage sections to `pyproject.toml`. Scope `source` to the integration package so coverage reflects first-party code only.

Add the dependency with uv (keeps `uv.lock` in sync):
```bash
uv add --dev pytest-cov
```

Then add the coverage config to `pyproject.toml`:
```toml
[tool.coverage.run]
branch = true
source = ["custom_components/llm_home_controller"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "\\.\\.\\.",
]
```

Notes on the chosen values:
- `fail_under = 80` per the recommendation in this finding. Before committing, run `uv run pytest --cov` once and confirm the suite already clears 80; if current coverage is higher, raise the floor to just below the measured value so the gate is meaningful but not immediately red. Treat the number as a floor, not a target.
- The `\\.\\.\\.` (ellipsis) exclusion covers the Protocol/abstract bodies in `providers/__init__.py`.
- Do **not** add `[tool.pytest.ini_options] addopts = "--cov"` unless the team wants coverage on every bare `pytest` run; PY-035 only requires the config to exist so `pytest --cov` reads it. Leaving `addopts` untouched keeps fast iterative test runs available. (Optional, team preference.)

## Acceptance criteria
- [ ] `pytest-cov` appears in the `dev` group of `pyproject.toml` and in `uv.lock`.
- [ ] `pyproject.toml` contains `[tool.coverage.run]` with `branch = true` and `source = ["custom_components/llm_home_controller"]`.
- [ ] `pyproject.toml` contains `[tool.coverage.report]` with `fail_under` set, `show_missing = true`, and `exclude_lines` including `if TYPE_CHECKING:`, `raise NotImplementedError`, `pragma: no cover`, and the ellipsis pattern.
- [ ] `uv run pytest --cov` runs, prints a per-file missing-lines report, and exits 0 (the suite meets the configured `fail_under`).
- [ ] `uv run pytest --cov --cov-fail-under=999` exits non-zero, proving the gate is wired to fail rather than being inert.
- [ ] `uv run ruff check .` still passes (config-only change should not affect lint).

## Notes / risks
- Verify the actual coverage number before locking `fail_under`. If the suite currently sits below 80, do not lower the gate to pass — instead file a follow-up to cover the gap, or set the floor at the current value with a TODO to ratchet up. Shipping a gate the suite cannot pass blocks all future work.
- HA custom-component test runs can be slow; `--cov` adds tracing overhead. If iterative runs become painful, leave `addopts` clean (as above) so `--cov` is opt-in locally and only forced in CI.
- Related: PY-035 also suggests `--cov-report=xml --cov-report=term-missing` in CI. This repo has no CI workflow in scope for this ticket; if/when CI is added, wire the same config there — out of scope here.
- No source-code changes required; if a future `if TYPE_CHECKING:` block or `@abstractmethod` is added, the exclusion list already accounts for it.
