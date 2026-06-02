# BP-PY-03: Add `--strict-markers` and `--strict-config` to pytest addopts

> **Status: DONE** — shipped in commit `0582f6e`. `addopts = ["--strict-markers", "--strict-config"]` added; 228 tests pass.

**Severity:** medium  |  **Rule:** PY-030 (python-best-practices)  |  **Area:** `pyproject.toml` — `[tool.pytest.ini_options]` (test configuration)

## Context
The pytest configuration in `pyproject.toml` declares only `asyncio_mode` and `testpaths`. It has no `addopts` line, so neither `--strict-markers` nor `--strict-config` is in effect.

Without `--strict-markers`, a typo'd marker decorator (e.g. `@pytest.mark.asyncoi` instead of `@pytest.mark.asyncio`) is silently accepted by pytest rather than failing. The mistyped marker simply does nothing — for an async test that means it would no longer be awaited correctly and could silently pass without actually running its assertions. The suite leans heavily on this exact marker: there are ~50+ `@pytest.mark.asyncio` decorators across `tests/test_entity.py`, `tests/test_memory.py`, `tests/test_ai_task.py`, etc. (the suite also runs `asyncio_mode = "auto"`, but explicit markers are still used throughout). A single typo in any of them is currently undetectable.

Without `--strict-config`, a typo in a pytest config key (or an unknown key from a removed/renamed plugin option) is silently ignored instead of erroring, so a misconfigured run looks healthy.

This is a preventative hardening change. The current suite uses only the built-in `asyncio` marker (registered by the `pytest-asyncio` plugin) and no custom config keys, so enabling strict mode does not break anything today.

## Current state
`pyproject.toml:24-26`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

Marker usage (sampled via `grep -rn "pytest.mark" tests/`): only `@pytest.mark.asyncio` appears — the built-in marker registered by `pytest-asyncio`. No custom markers are defined or used.

## Proposed fix
Add an `addopts` list with the two strict flags. Minimal, scoped change to the existing table:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = ["--strict-markers", "--strict-config"]
```

Do NOT add a `markers = [...]` table — there are no custom markers in this repo, and the `asyncio` marker is registered by the plugin, so it satisfies `--strict-markers` without manual registration. Keep the change to these two flags only (do not add `-q`, `--tb=short`, `filterwarnings`, or `minversion` — those are separate findings/decisions, out of scope here).

## Acceptance criteria
- [ ] `pyproject.toml` `[tool.pytest.ini_options]` contains `addopts = ["--strict-markers", "--strict-config"]`.
- [ ] `uv run pytest tests/ --collect-only -q` collects the same set of tests as before the change with no new marker/config errors (note: a pre-existing `turbojpeg` import error in `tests/test_ai_task.py` collection is unrelated to this ticket — see Notes).
- [ ] Introducing a deliberate typo'd marker (e.g. `@pytest.mark.asyncoi`) now causes pytest to fail with "unknown marker" rather than silently passing; revert the typo after confirming.
- [ ] No `markers` table was added and no other pytest options were introduced.

## Notes / risks
- Pre-existing, unrelated: running `uv run pytest tests/ --collect-only` currently fails collection on `tests/test_ai_task.py` with `ModuleNotFoundError: No module named 'turbojpeg'` (an environment/dependency issue, not a config one). This ticket does not address that; use the other test files (or fix the env separately) to validate the strict-marker behavior.
- Low risk: the only marker in use (`asyncio`) is plugin-registered, so `--strict-markers` will not flag it. `--strict-config` only errors on unknown config keys, of which there are none.
- The repo pins `pytest>=8.0` (`pyproject.toml:18`), so the `[tool.pytest.ini_options]` back-compat table and list-form `addopts` are correct. The pytest 9.0+ native `[tool.pytest]` / `strict_markers = true` form is NOT applicable here and should not be used.
- Related rule context: PY-030 also recommends `filterwarnings = ["error", ...]` and `minversion`; those are intentionally excluded from this ticket to keep the change surgical.
