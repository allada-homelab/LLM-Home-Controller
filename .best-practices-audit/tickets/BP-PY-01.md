# BP-PY-01: Add mypy strict type checking to dev deps and CI

> **Status: DONE (2026-07-03).** Maintainer reversed the earlier WON'T DO decision and
> adopted a type checker — but **basedpyright** (matching the user's global Python default),
> not the originally-proposed mypy strict. Configured in `pyrightconfig.json`:
> `typeCheckingMode = "standard"` (non-strict), `pythonVersion = "3.13"` (the
> `requires-python` floor), `reportMissingTypeStubs = "none"`, scoped to
> `custom_components/llm_home_controller` + `tests` with `reportPrivateUsage` /
> `reportUnusedFunction` relaxed for tests. Added `basedpyright` to the dev group
> (recorded in `uv.lock`) and a `just typecheck` recipe. CI runs `uv run basedpyright`
> in `lint.yml` with `continue-on-error: true` initially (type baseline not yet clean).
> The mypy-strict proposal below is superseded and kept for history.

**Severity:** medium  |  **Rule:** PY-012 (python-best-practices)  |  **Area:** `pyproject.toml`, `.github/workflows/lint.yml`, `custom_components/llm_home_controller/`

## Context
The codebase is heavily typed (type aliases, annotated method signatures, and even
manual `# type: ignore[...]` suppressions), but **no static type checker is configured
or run anywhere**. There is no `[tool.mypy]` or `[tool.pyright]` section in
`pyproject.toml`, no `mypy`/`pyright`/`basedpyright` in the dev dependency group, and
CI runs only `ruff check` + `ruff format --check`. Nothing validates the type
annotations.

Concretely, this means the four existing suppressions in the code are unverified — they
could already be stale (the underlying error gone) and no one would know:

- `custom_components/llm_home_controller/entity.py:328` — `raise last_error  # type: ignore[misc]`
- `custom_components/llm_home_controller/entity.py:368` — `raise first_error  # type: ignore[misc]`
- `custom_components/llm_home_controller/conversation.py:223` — `system = chat_log.content[0].content  # type: ignore[union-attr]`
- `custom_components/llm_home_controller/conversation.py:261` — `system = chat_log.content[0].content  # type: ignore[union-attr]`

Without a type checker, refactors of this triple-inheritance conversation entity and the
fallback-model loop in `entity.py` lose the safety net that the type hints were written
to provide. mypy `strict` mode (which includes `warn_unused_ignores`) would both catch
real type bugs and flag any `# type: ignore` that no longer suppresses anything.

## Current state
`pyproject.toml` (lines 13-22) — dev group has no type checker:
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
```
No `[tool.mypy]` section exists anywhere in `pyproject.toml` (file ends at line 47 with
the ruff isort config).

`.github/workflows/lint.yml` (lines 9-20) — Lint job runs ruff only, no type check step:
```yaml
jobs:
  ruff:
    name: Ruff
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .
```

## Proposed fix
Add mypy in strict mode (rule recommendation for an established, already-typed codebase),
add it to the dev group, and run it in CI. The project uses uv (`uv.lock` is committed),
so the CI step should use `uv run`.

1. **`pyproject.toml`** — add `mypy` to the dev group:
```toml
[dependency-groups]
dev = [
    "hassil>=3.0",
    "home-assistant-intents>=2026.2.3",
    "litellm<=1.82.6",
    "mypy>=1.9",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-homeassistant-custom-component",
    "ruff>=0.9",
]
```

2. **`pyproject.toml`** — add a `[tool.mypy]` section. Strict mode, pinned to the
   project's Python version. Scope around HA-core stub gaps with a per-module override
   rather than disabling checks globally:
```toml
[tool.mypy]
python_version = "3.13"
strict = true

# HomeAssistant and its test/runtime deps ship without complete type stubs.
# Scope the gap narrowly so first-party code stays fully strict.
[[tool.mypy.overrides]]
module = [
    "homeassistant.*",
    "hassil.*",
    "home_assistant_intents.*",
]
ignore_missing_imports = true
```

3. **`.github/workflows/lint.yml`** — add a type-check step to the existing `ruff` job
   (or a sibling job). Using `uv` to honor the lockfile and dev group:
```yaml
      - run: ruff check .
      - run: ruff format --check .
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --locked
      - run: uv run mypy custom_components
```
   (If the maintainer prefers keeping the ruff job pip-only, add a separate `mypy` job
   that does `uv sync --locked` then `uv run mypy custom_components`.)

4. Run `uv lock` after editing `pyproject.toml` so `uv.lock` includes mypy, then
   run `uv run mypy custom_components` locally and resolve whatever it surfaces. Expect
   the run to flag any of the four `# type: ignore` lines above that are now stale
   (`warn_unused_ignores`), plus possibly genuine errors that strict mode reveals for the
   first time. Fixing those errors is part of this ticket; do not silence them with new
   blanket ignores.

## Acceptance criteria
- [ ] `mypy` is present in the `[dependency-groups].dev` list in `pyproject.toml` and recorded in `uv.lock`.
- [ ] `pyproject.toml` has a `[tool.mypy]` section with `python_version = "3.13"` and `strict = true`.
- [ ] HA-core / intent stub gaps are scoped via a `[[tool.mypy.overrides]]` block with `ignore_missing_imports = true` (not by weakening top-level strict settings).
- [ ] `.github/workflows/lint.yml` runs `uv run mypy custom_components` (via `uv sync --locked`) and the workflow fails if mypy reports errors.
- [ ] `uv run mypy custom_components` passes cleanly on the branch (zero errors).
- [ ] Each of the four existing `# type: ignore[...]` comments is either confirmed still-needed (mypy errors without it) or removed if `warn_unused_ignores` flags it as stale.

## Notes / risks
- Strict mode on a previously-unchecked codebase commonly surfaces a batch of pre-existing
  errors. If the count is large, it is acceptable to land the config + CI step first using
  a temporary per-module relaxation, but the goal of this ticket is a clean strict pass —
  do not leave broad `disable_error_code` escape hatches.
- Related rule: PY-013 (specific `# type: ignore[code]` with reason) — the four existing
  suppressions already use specific error codes; verify they still apply once mypy runs.
- The chosen checker is **mypy** per the rule recommendation for established typed
  codebases. The user's global Python default is `basedpyright`; mypy is preferred here
  because the rule explicitly recommends it for this situation and `strict`'s
  `warn_unused_ignores` directly addresses the stale-suppression evidence. Flag to the
  maintainer if they'd rather standardize on basedpyright (`typeCheckingMode = "strict"`).
- `tests/` is intentionally out of scope for the CI step (`mypy custom_components` only);
  widening to tests can be a follow-up and would likely need pytest-fixture typing work.
- The `astral-sh/setup-uv@v5` action pin should be checked against the latest major at
  implementation time.
