# BP-PY-06: Add per-file ruff ignores for tests and `__init__.py`

**Severity:** low  |  **Rule:** PY-022 (python-best-practices)  |  **Area:** `pyproject.toml` ruff lint config; `tests/`, `**/__init__.py`

## Context
The project's ruff config (`pyproject.toml`) has no `[tool.ruff.lint.per-file-ignores]`
section. Today this is harmless because the lint `select` list does NOT include the
`S` (flake8-bandit) family — so `S101` (use of `assert`) never fires. `uv run ruff
check tests/` currently passes clean.

The finding is **latent, not active**. It becomes a real problem the moment the `S`
family is enabled (the recommendation under sibling ticket PY-021 proposes exactly
that). The `tests/` tree contains ~460 `assert` statements across ~5,685 lines; every
one would trip `S101`. Without a per-file ignore, enabling `S` either floods the lint
output with hundreds of false positives or forces a global `ignore = ["S101"]`, which
would also silence the rule in production code — defeating the point of enabling
bandit (a stray `assert user.is_admin` in a handler is a real bug because Python's `-O`
strips asserts).

The correct mechanism is a per-file ignore: keep `S101`/`S105`/`S106`/`S107` strict in
production code, relaxed only in `tests/`. The `F401` (unused import) ignore for
`__init__.py` is preemptive housekeeping: the two package `__init__.py` files
(`custom_components/llm_home_controller/__init__.py`,
`custom_components/llm_home_controller/providers/__init__.py`) do not currently rely on
re-exports that would trip `F401`, but the pattern is cheap insurance and matches the
rule's recommendation.

## Current state
`/workspace/LLM-Home-Controller/pyproject.toml` lines 28–47 — the entire ruff config.
There is no `per-file-ignores` table anywhere in the file, and `S` is absent from
`select`:

```toml
[tool.ruff]
target-version = "py313"
line-length = 120

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "SIM",    # flake8-simplify
    "TID",    # flake8-tidy-imports
    "RUF",    # ruff-specific rules
]

[tool.ruff.lint.isort]
known-first-party = ["custom_components.llm_home_controller"]
```

## Proposed fix
Add a `[tool.ruff.lint.per-file-ignores]` table to `pyproject.toml`. Place it inside
the `[tool.ruff.lint]` block (e.g. directly after the `select` list, before
`[tool.ruff.lint.isort]`). This is a config-only change — no source files are touched.

Add:

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",  # asserts are pytest's primary assertion mechanism
    "S105",  # hardcoded "password" strings in test fixtures are fine
    "S106",  # same — test passwords passed as args
    "S107",  # same — test passwords as defaults
]
"**/__init__.py" = [
    "F401",  # re-exports look unused to Pyflakes
]
```

Scope note: the recommendation lands this alongside the PY-021 change that enables the
`S` family. This ticket covers ONLY the per-file-ignores table. Adding it before `S`
is selected is a no-op on current lint output (the ignored rules are not active yet),
so it is safe to merge independently or together with the PY-021 ticket.

## Acceptance criteria
- [ ] `[tool.ruff.lint.per-file-ignores]` exists in `pyproject.toml` with the
      `"tests/**/*.py"` and `"**/__init__.py"` patterns above.
- [ ] `uv run ruff check .` still passes (exit 0) with the new config in place.
- [ ] No source files (`.py`) are modified by this ticket — config change only.
- [ ] If/when the `S` family is added to `select` (PY-021), `uv run ruff check tests/`
      does NOT report `S101` violations, confirming the ignore is wired correctly.

## Notes / risks
- Ordering: pairs with the PY-021 ticket (enable `S` family). This ticket is the
  prerequisite that prevents the ~460 `S101` hits PY-021 would otherwise cause. Land
  this first, or simultaneously, with PY-021.
- The rule reference also suggests `F403` for `__init__.py`; omitted here because the
  project uses no wildcard re-exports (`from .x import *`) and the recommendation
  scoped this finding to `F401` only. Add `F403` only if a future `__init__.py`
  introduces a star import.
- Glob `"**/__init__.py"` matches both the component package and the `providers/`
  subpackage `__init__.py` files. Ruff resolves these globs relative to the
  `pyproject.toml` location, which is the repo root — so the patterns are correct as
  written.
- Verified latent, not active: `uv run ruff check tests/` currently prints
  "All checks passed!" because `S` is not selected.
