# BP-UV-05: Align the runtime Python pin so .python-version, CI, and the ruff target agree

> **Status: DONE (with deviation)** — shipped in commit `0582f6e`. Runtime drift fixed: `setup-uv` now reads `.python-version` (3.14) so CI no longer hardcodes 3.13. **Deviation:** ruff `target-version` was kept at `py313` (the `requires-python` floor), NOT bumped to `py314` as this ticket proposed — linting at the minimum supported version is correct, and `py314` triggers a ruff 0.15.1 formatter bug that rewrites valid `except (A, B):` into invalid syntax.

**Severity:** medium  |  **Rule:** UVP-050 (uv-best-practices)  |  **Area:** `.python-version`, `.github/workflows/lint.yml`, `pyproject.toml`, `.devcontainer/devcontainer.json`

## Context
The repo currently declares the development/runtime Python version in three places that disagree, so there is no single source of truth for "which interpreter do we actually run against."

- `.python-version` pins `3.14` (the runtime uv selects locally and in dev).
- `.devcontainer/devcontainer.json` installs Python `3.14` — so the actual dev container matches `.python-version`.
- The lint CI job (`lint.yml`) provisions `3.13`.
- ruff's `target-version` is `py313`.

`requires-python = ">=3.13"` in `pyproject.toml` is the compatibility *floor* and is correct as-is per UVP-050 (floor in `requires-python`, exact pin in `.python-version`) — it is not the source of the drift and should not change.

Why it matters: developers run and lint against 3.14 locally, but CI lints against 3.13 with `target-version = py313`. ruff's `UP`/pyupgrade rules and any version-gated lint behavior are evaluated against 3.13 in CI while contributors see 3.14 results locally — a recipe for "passes locally, surprises in CI" (and vice versa). It also means the codebase is never linted with 3.14 as the target even though that is the declared runtime. `uv.lock` already forks resolution markers across `< 3.13.2`, `>= 3.13.2 and < 3.14`, and `>= 3.14`, so both interpreter lines are live in the lockfile and the ambiguity is real, not theoretical.

## Current state

`.python-version` (entire file):
```
3.14
```

`.devcontainer/devcontainer.json:14-16`:
```json
"ghcr.io/devcontainers/features/python:1": {
    "version": "3.14"
}
```

`.github/workflows/lint.yml:11-21` (uses `actions/setup-python`, NOT `setup-uv`, so it does not auto-read `.python-version`):
```yaml
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

`pyproject.toml:28-30`:
```toml
[tool.ruff]
target-version = "py313"
line-length = 120
```

`pyproject.toml:6` (the compat floor — leave unchanged):
```toml
requires-python = ">=3.13"
```

## Proposed fix
Pick the runtime that dev actually uses (3.14) as the one source of truth and bring CI + the ruff target up to it. This is the minimal change because `.python-version` and the dev container already agree on 3.14; only CI and the ruff target are stragglers.

`.github/workflows/lint.yml` — before/after:
```yaml
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"   # before
          python-version: "3.14"   # after
```

`pyproject.toml`:
```toml
target-version = "py313"   # before
target-version = "py314"   # after
```

Leave `requires-python = ">=3.13"` untouched — the project still *supports* 3.13 for downstream consumers; we just develop/lint against 3.14.

Do NOT change `.python-version` or the dev container; they are already the intended runtime.

Alternative (not recommended): move `.python-version` and the dev container down to `3.13` to match CI. Rejected because it downgrades the actual dev runtime to chase the stragglers, and 3.14 is already in use locally.

Future improvement (out of scope for this ticket, tracked separately if desired): migrate `lint.yml` to `astral-sh/setup-uv` so it reads `.python-version` automatically and `uv run ruff ...`, eliminating the hardcoded `python-version` line entirely. Not done here to keep the change surgical.

## Acceptance criteria
- [ ] `lint.yml` sets `python-version: "3.14"`.
- [ ] `pyproject.toml` sets `target-version = "py314"`.
- [ ] `.python-version` still reads `3.14` and the dev container Python feature still pins `3.14` (unchanged).
- [ ] `requires-python = ">=3.13"` is unchanged.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass against the 3.14 target (run locally before pushing; CI will mirror this).
- [ ] No source of the runtime pin disagrees: `.python-version`, dev container, CI `setup-python`, and ruff `target-version` all reference the 3.14 line.

## Notes / risks
- ruff `target-version = "py314"` may surface additional `UP` (pyupgrade) suggestions that were dormant under py313. Run `uv run ruff check .` after the change and fix or review any new findings before merging — these are part of completing this ticket, not a separate one.
- Related rule UVP-051: keep `.python-version` pinned to the *minor* (`3.14`), never a patch (`3.14.x`). It already is — do not "tighten" it.
- `lint.yml` uses `actions/setup-python`, not `astral-sh/setup-uv`, so bumping the version must be done manually in the workflow; it will not pick up `.python-version` on its own today.
- No code in `custom_components/` or `tests/` depends on a 3.14-only feature, so bumping the CI runner to 3.14 should not change test behavior — but the lint workflow is the only one touched here; `validate.yml` (hassfest) has no Python pin and is unaffected.
