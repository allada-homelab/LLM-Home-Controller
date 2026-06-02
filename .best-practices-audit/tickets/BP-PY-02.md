# BP-PY-02: Add the `S` (flake8-bandit) ruff rule family with a tests per-file-ignore

**Severity:** medium  |  **Rule:** PY-021 (python-best-practices)  |  **Area:** `pyproject.toml` ruff lint config (whole-repo static security linting)

## Context
The ruff lint selection in `pyproject.toml` enables `E, W, F, I, UP, B, SIM, TID, RUF` but omits `S` (flake8-bandit), the security-focused family. PY-021 lists `S` as part of the recommended baseline precisely because it flags a class of real security footguns: `eval`/`exec`, `pickle.loads`, `subprocess(..., shell=True)`, `requests` calls without timeouts, hardcoded passwords/tokens, weak hashing, and insecure temp-file usage.

This omission matters for this integration specifically because of what it handles:
- API keys and auth credentials (parent config entry stores API URL + key).
- User-supplied HTTP headers and request bodies sent to arbitrary OpenAI-compatible backends.
- Arbitrary JSON tool/parameter schemas coming back from an LLM.
- Dynamic Home Assistant service dispatch driven by LLM tool calls.

Any of those is exactly the kind of surface where a future change could introduce, e.g., a credential logged in plaintext, a timeout-less outbound call, or an unsafe deserialization — the bandit family catches these statically in CI before they ship.

I verified the fix is low-risk and well-scoped:
- Running `ruff check custom_components --select S` today reports **All checks passed** — adding `S` introduces **zero** new violations in production source.
- Running `ruff check tests --select S` reports **460 S101 violations**, all of them legitimate pytest `assert` statements. `S101` (and the test-password rules `S105`/`S106`/`S107`) must be ignored for the `tests/` tree, exactly as the recommendation states.
- `S101` is the *only* S code that fires anywhere in the repo today; `S105`/`S106`/`S107` do not currently fire but are included in the test ignore pre-emptively (PY-022) so future test fixtures with dummy passwords/tokens don't trip the rule.

## Current state
`pyproject.toml:32-43` — the lint selection, missing `S`:
```toml
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
```
There is currently **no** `[tool.ruff.lint.per-file-ignores]` section anywhere in `pyproject.toml`. The only ruff lint subsection present is `[tool.ruff.lint.isort]` at lines 45-46.

ruff version in use: `0.15.1` (dev dependency pinned `ruff>=0.9`, line 21).

## Proposed fix
Two minimal edits to `pyproject.toml`, nothing else.

1. Add `"S"` to the `select` list (place it after `"UP"` to match the rule-doc ordering; any position works):
```toml
    "UP",     # pyupgrade
    "S",      # flake8-bandit (security)
    "B",      # flake8-bugbear
```

2. Add a new `per-file-ignores` section relaxing the test-only security rules. Put it adjacent to the existing `[tool.ruff.lint.isort]` block:
```toml
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",   # asserts are pytest's primary mechanism
    "S105",   # hardcoded test secrets in fixtures are OK
    "S106",   # same — test secrets passed as kwargs
    "S107",   # same — test secrets as defaults
]
```

Do **not** add a global `ignore = ["S101"]` — that would silence the assert check in production code too, where `-O`-stripped asserts are a genuine concern (PY-022). Keep the ignore scoped to `tests/**/*.py`.

No source-code changes are required: production source already passes the full `S` family.

## Acceptance criteria
- [ ] `pyproject.toml` `[tool.ruff.lint].select` includes `"S"`.
- [ ] `pyproject.toml` has a `[tool.ruff.lint.per-file-ignores]` section with `"tests/**/*.py"` ignoring at least `S101`, `S105`, `S106`, `S107`.
- [ ] No global `ignore` entry was added for `S101` (the check stays active in `custom_components/`).
- [ ] `uv run ruff check .` passes with **zero** errors after the change.
- [ ] `uv run ruff check tests --select S` would still flag a stray `eval`/`pickle`-class issue (i.e. only the assert/password codes are ignored, not the whole `S` family) — confirm by checking the ignore list contains specific codes, not `"S"`.

## Notes / risks
- Pairs directly with PY-022 (per-file ignores for tests). This is the canonical example from that rule.
- Low blast radius: verified zero new violations in `custom_components/`, so this will not block CI on existing code.
- If a future PR genuinely needs a security exception in production source (e.g. an intentional `subprocess` call), prefer a line-level `# noqa: Sxxx` with a justification comment over widening the ignore list — and note `RUF100` (already enabled via the `RUF` family) will flag the noqa if it ever becomes stale (PY-025).
- `S105`/`S106`/`S107` do not fire today; they are included pre-emptively per PY-022 so dummy-credential test fixtures added later don't trip the rule. Harmless if they never fire.
- This is an additive lint-config change only — no runtime behavior changes, no source edits.
