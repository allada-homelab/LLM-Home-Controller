# Python Best-Practices Audit — `llm-home-controller`

Audited the Python surface of this Home Assistant custom integration against the
`python-best-practices` (PY-*) rule family. Files reviewed: `pyproject.toml`
(outside `[tool.uv]`/`[dependency-groups]`), `.python-version`, the `[tool.ruff]`
and `[tool.pytest.ini_options]` config, `.github/workflows/lint.yml` and
`validate.yml`, all of `custom_components/llm_home_controller/**/*.py`
(`__init__.py`, `entity.py`, `conversation.py`, `config_flow.py`, `const.py`,
`ai_task.py`, `memory.py`, `sensor.py`, `entity_tools.py`, `providers/*.py`), and
`tests/**/*.py` + `conftest.py`.

The code itself is in good shape on the PY rules that touch source files: every
module uses `logger = logging.getLogger(__name__)` (PY-050) with positional log
args and no f-strings (PY-051); `_LOGGER.exception(...)` is used only inside
`except` blocks (PY-054); all `# type: ignore[...]` / `# noqa: ...` carry explicit
codes (PY-016); modern typing syntax is used throughout (`X | None`, `list[str]`,
`type` aliases, PEP 695) (PY-010/PY-071); `Protocol` is used for the provider
interface (PY-013); and the streaming async generators are drained to exhaustion
under `try/finally: response.release()`, so PY-044 (`aclosing`) genuinely does not
apply. No naive `datetime` usage exists anywhere, so PY-073/DTZ don't apply.

Findings: 0 high, 4 medium, 2 low.

| Rule ID | Severity | Title | Location |
|---------|----------|-------|----------|
| PY-012 / PY-011 | medium | No static type checker configured or run in CI | `pyproject.toml`, `.github/workflows/lint.yml` |
| PY-021 | medium | ruff rule selection omits `S` (security/bandit) | `pyproject.toml:30` |
| PY-030 | medium | pytest config missing `addopts = --strict-markers --strict-config` | `pyproject.toml:24` |
| PY-035 | medium | No coverage gate (`fail_under`, `branch = true`) | `pyproject.toml` |
| PY-004 | low | No `[build-system]` table — falls back to legacy setuptools | `pyproject.toml` |
| PY-022 | low | No `[tool.ruff.lint.per-file-ignores]` for tests/`__init__.py` | `pyproject.toml` |

---

## PY-012 / PY-011 — No static type checker configured or run in CI (medium)

**Location:** `pyproject.toml` (no `[tool.mypy]`/`[tool.pyright]`), `.github/workflows/lint.yml`

**Evidence.** No type-checker config exists anywhere and none runs in CI:

```
$ grep -rn "tool.mypy|tool.pyright|tool.basedpyright" pyproject.toml  → (none)
$ ls mypy.ini .mypy.ini pyrightconfig.json                           → no files
$ grep -rn "mypy|pyright" pyproject.toml .github/workflows/          → no type checker in deps/CI
```

`lint.yml` runs only `ruff check .` and `ruff format --check .`. The dev
dependency group lists `ruff` but no type checker.

**Why it matters.** The codebase is heavily typed (PEP 604/585/695 annotations,
`Protocol` providers, a `type` alias) but nothing verifies those annotations.
Ruff is not a type checker (PY-020). Without a checker the type hints document
intent but never catch a bug — e.g. the several `# type: ignore[union-attr]` /
`# type: ignore[misc]` suppressions in `conversation.py`/`entity.py` can silently
go stale because no `warn_unused_ignores` ever flags them. Home Assistant core
itself runs strict mypy on integrations, so this is meaningful hardening for a
HA component.

**Recommendation.** Add a type checker in strict mode and run it in CI. For this
established, already-strictly-typed codebase, mypy is the lower-friction choice
(PY-011):

```toml
[tool.mypy]
python_version = "3.13"
strict = true
warn_unused_configs = true
```

Add `mypy` (or `basedpyright`) to the `dev` dependency group and a CI step
`uv run mypy custom_components`. Expect to scope a few HA-core-stub gaps via
`[[tool.mypy.overrides]] ignore_missing_imports = true`.

---

## PY-021 — ruff rule selection omits `S` (security/bandit) (medium)

**Location:** `pyproject.toml:30` (`[tool.ruff.lint].select`)

**Evidence.**

```toml
select = [
    "E", "W", "F", "I", "UP", "B", "SIM", "TID", "RUF",
]
```

The recommended baseline is `E, W, F, B, I, UP, S, SIM, RUF`. `S`
(flake8-bandit) is absent.

**Why it matters.** This integration handles secrets and untrusted input: it
reads API keys and user-supplied custom HTTP headers, parses arbitrary
user-provided JSON tool/parameter schemas, and dynamically dispatches Home
Assistant service calls from LLM-driven tool definitions (`entity.py`,
`config_flow.py`). The `S` family flags exactly the footguns relevant here
(unsafe deserialization, hardcoded credentials, etc.). Of the recommended set,
`S` is the one with clear security upside that is currently missing.

**Recommendation.** Add `"S"` to `select`. Because asserts in tests would then
trip `S101`, pair it with a per-file ignore (see PY-022 below):

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "S105", "S106", "S107"]
```

`TID` (currently selected) is a fine project addition; keep it.

---

## PY-030 — pytest config missing `--strict-markers` / `--strict-config` (medium)

**Location:** `pyproject.toml:24` (`[tool.pytest.ini_options]`)

**Evidence.**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

No `addopts` line; `--strict-markers` and `--strict-config` are absent.

**Why it matters.** `--strict-markers` is the single highest-value pytest flag:
without it, a typo'd marker (`@pytest.mark.integartion`) silently does not filter
and the test runs in every suite, only emitting a warning nobody reads.
`--strict-config` turns config typos into errors. The suite currently only uses
the built-in `@pytest.mark.asyncio` (no custom markers yet), so this is
preventative hardening rather than a live bug — but it is cheap and stops the
class of error before custom markers are introduced.

**Recommendation.**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = ["--strict-markers", "--strict-config"]
```

---

## PY-035 — No coverage gate (medium)

**Location:** `pyproject.toml` (no `[tool.coverage.*]` sections)

**Evidence.**

```
$ grep -n "coverage|fail_under|pytest-cov" pyproject.toml  → (none)
```

There is no `[tool.coverage.run]`/`[tool.coverage.report]` config, no
`branch = true`, no `fail_under`, and `pytest-cov` is not a dev dependency.

**Why it matters.** The project has a substantial, mature test surface (11 test
modules, `tests/test_entity.py` alone is ~1,300 lines) and the CLAUDE.md mandates
testing every error path. Without a branch-coverage gate, regressions in the
error-handling paths (connection refused, auth failure, malformed SSE, JSON
parse failures) can silently lose coverage. This is past the "<100 lines, skip
it" exception in the rule.

**Recommendation.** Add `pytest-cov` to the `dev` group and a coverage gate:

```toml
[tool.coverage.run]
branch = true
source = ["custom_components/llm_home_controller"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:", "\\.\\.\\."]
```

Treat 80% as a starting floor, not a target.

---

## PY-004 — No `[build-system]` table; legacy setuptools fallback (low)

**Location:** `pyproject.toml` (no `[build-system]` section)

**Evidence.**

```
$ grep -c "build-system" pyproject.toml  → 0
```

A stale `llm_home_controller.egg-info/` directory (gitignored, not tracked)
exists from a prior editable install, confirming the project was built once via
the legacy setuptools auto-fallback — `top_level.txt` lists
`custom_components`, `homeassistant`, `aiohttp`, `voluptuous` (over-broad
auto-discovery, exactly the PY-004 failure mode).

**Why it matters.** With no `[build-system]` table, any `pip install -e .` /
`uv pip install -e .` falls back to legacy setuptools with a deprecation warning,
and auto-discovery picks up the wrong top-level packages. This is a HACS-/symlink-
distributed HA integration that is not published to PyPI, so the impact is low —
but the project *is* declared as installable (`[project]` with `dependencies`,
and it has been editable-installed). Either commit to it being installable or
not.

**Recommendation.** If the package is meant to be pip-installable (the dev
container symlinks/installs it), declare a backend explicitly and scope
discovery:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["custom_components/llm_home_controller"]
```

If it is *only* ever consumed as a HA custom component (never installed as a
distribution), that is a legitimate "When NOT to apply" case — but then the
`egg-info` artifact should be removed and an editable install avoided.

---

## PY-022 — No per-file ignores for tests / `__init__.py` (low)

**Location:** `pyproject.toml` (no `[tool.ruff.lint.per-file-ignores]`)

**Evidence.**

```
$ grep -n "per-file-ignores" pyproject.toml  → (none)
```

**Why it matters.** This is currently latent: with the present `select` list
there is no rule that needs relaxing per-file. It becomes a real requirement the
moment `S` is enabled (PY-021) — `S101` would then fire on every pytest `assert`
across the ~3,000 lines of tests. Recording it here so the PY-021 fix lands
complete rather than breaking the test lint.

**Recommendation.** Add alongside the PY-021 change:

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "S105", "S106", "S107"]
"**/__init__.py" = ["F401"]
```
