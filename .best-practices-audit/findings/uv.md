# uv best-practices audit — LLM-Home-Controller

Audited the uv surface of the repo: `pyproject.toml` (`[project]`, `[dependency-groups]`, `requires-python`, ruff/pytest config), `.python-version`, the committed `uv.lock`, `.gitignore`, and both GitHub Actions workflows (`.github/workflows/lint.yml`, `validate.yml`). The project is a genuine uv project — `uv.lock` is committed (UVP-010 ✅), PEP 735 `[dependency-groups]` is used instead of the deprecated `[tool.uv.dev-dependencies]` (UVP-001 ✅), `requires-python` is a floor (UVP-003 ✅), and `.python-version` pins a minor not a patch (UVP-051 ✅). However, **CI does not use uv at all** — it installs tooling via raw `pip` and never validates the lockfile — and there is no `[tool.uv]` table to pin the resolver floor, plus a Python-version drift between the dev pin, the CI runner, and the lint target.

Findings: 1 high, 4 medium, 0 low.

| Rule ID | Severity | Title | Location |
|---|---|---|---|
| UVP-032 | high | CI never runs `uv sync --locked`; lockfile is never validated | `.github/workflows/lint.yml`, `validate.yml` |
| UVP-040 | medium | Project-context tool `ruff` installed via raw `pip` in CI, bypassing the lockfile | `.github/workflows/lint.yml:18` |
| UVP-008 | medium | No `[tool.uv] required-version` to pin a resolver floor | `pyproject.toml` (no `[tool.uv]`) |
| UVP-028 | medium | No `uv audit` step in CI to scan `uv.lock` for known CVEs | `.github/workflows/*.yml` |
| UVP-050 | medium | `.python-version` (3.14) drifts from CI runner / ruff target (3.13) | `.python-version`, `lint.yml:17`, `pyproject.toml:29` |

---

## UVP-032 — CI never runs `uv sync --locked`; the committed lockfile is never validated (high)

**Location:** `.github/workflows/lint.yml`, `.github/workflows/validate.yml`

```yaml
# lint.yml
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install ruff
      - run: ruff check .
```

```yaml
# validate.yml — only hassfest, no Python env / uv at all
      - uses: home-assistant/actions/hassfest@master
```

**Why it matters.** This is a committed-uv-project (`uv.lock` + `[dependency-groups]` are tracked in git) whose CI never touches uv. Nothing in CI runs `uv sync --locked` or `uv lock --check`, so the single most important guard UVP-032 describes is absent: a PR that edits `pyproject.toml` (e.g. bumps `litellm` or `pytest`) without re-running `uv lock` will sail through CI because CI never reads the lockfile. The stale lockfile then breaks the next contributor's `uv sync --locked` locally, and any deployment that does sync from the lock ships versions that were never tested. There is also no test job at all — the dev group declares `pytest`/`pytest-homeassistant-custom-component` but no workflow runs them.

**Recommended change.** Add a uv-based test/lint job that gates on the lockfile:

```yaml
      - uses: astral-sh/setup-uv@<sha> # v8.x, enable-cache: true (UVP-030/031)
      - run: uv lock --check          # pre-gate: lockfile current vs pyproject
      - run: uv sync --locked --dev   # install from the committed lock, fail if stale
      - run: uv run ruff check .
      - run: uv run pytest
```

---

## UVP-040 — `ruff` (a project-context tool in the dev group) is installed via raw `pip install ruff` in CI (medium)

**Location:** `.github/workflows/lint.yml:18`

```yaml
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .
```

```toml
# pyproject.toml — ruff is already declared as a dev dependency
[dependency-groups]
dev = [ ..., "ruff>=0.9", ]
```

**Why it matters.** `ruff` is a project-context tool already pinned in `[dependency-groups]` and the lockfile. CI ignores that and installs an unpinned, latest-from-PyPI `ruff` via `pip` into an ad-hoc environment. The exact "works locally / drifts in CI" failure UVP-040 (and the version-pinning angle of UVP-041) warns about: a contributor's `uv run ruff` and CI's `ruff` can be different versions, so formatting/lint results diverge with no source change. It also means the lockfile-pinned `ruff>=0.9` version is never actually exercised in CI.

**Recommended change.** Drop `pip install ruff` and run the lockfile-pinned tool: `uv sync --locked --dev` then `uv run ruff check .` / `uv run ruff format --check .`. This reuses the version everyone develops against.

---

## UVP-008 — No `[tool.uv] required-version` to pin a resolver floor (medium)

**Location:** `pyproject.toml` — there is no `[tool.uv]` table at all.

```toml
# pyproject.toml has [project], [dependency-groups], [tool.pytest.ini_options],
# [tool.ruff*] — but no [tool.uv]
```

**Why it matters.** This project has multiple consumers (a named git author, GitHub Actions, the documented HA dev-container workflow), so version skew between whichever uv each one runs can silently produce different `uv.lock` output from the same `pyproject.toml` — diffs that look like "someone re-locked" but are actually two resolvers disagreeing. UVP-008 makes the floor machine-checkable.

**Recommended change.** Add a baseline floor and bump it as features are adopted:

```toml
[tool.uv]
required-version = ">=0.5.14"
```

---

## UVP-028 — No `uv audit` step in CI to scan the lockfile for known vulnerabilities (medium)

**Location:** `.github/workflows/lint.yml`, `.github/workflows/validate.yml` (no audit job exists)

**Why it matters.** This is a deployable artifact (a Home Assistant integration with networked dependencies — `aiohttp`, `homeassistant`, `litellm`). A pinned `uv.lock` drifts into vulnerability over time as CVEs are disclosed against locked versions, with no signal until an incident. `uv audit` reads `uv.lock` directly (no sync needed), queries the OSV database, and exits non-zero on findings — a cheap CI gate. The "when NOT to apply" exceptions (air-gapped CI, template-only repos) do not apply here.

**Recommended change.** Add an audit job (ideally also on a daily `schedule:` to surface newly-disclosed CVEs):

```yaml
      - uses: astral-sh/setup-uv@<sha>
      - run: uv audit --no-dev
```

---

## UVP-050 — `.python-version` (3.14) drifts from the CI runner and ruff target (3.13) (medium)

**Location:** `.python-version` (`3.14`), `.github/workflows/lint.yml:17` (`python-version: "3.13"`), `pyproject.toml:6` (`requires-python = ">=3.13"`), `pyproject.toml:29` (`target-version = "py313"`)

```
# .python-version
3.14
```
```yaml
# lint.yml
          python-version: "3.13"
```

**Why it matters.** `.python-version` pins the local/dev runtime to **3.14**, but CI lint runs on **3.13** and ruff targets **py313**. UVP-050 is precisely about not conflating these: developers run/format against 3.14 semantics while CI validates against 3.13, so a 3.14-only construct (or a lint rule that differs by target) can pass locally and fail — or silently differ — in CI. `requires-python = ">=3.13"` (the compat floor) is correct and not the issue; the drift is between the *runtime pin* and what CI actually exercises. (Note the lockfile already forks resolution across 3.13/3.14, so both are in scope.)

**Recommended change.** Make the dev pin and the CI/lint target agree. Either move CI and `target-version` to `py314` to match `.python-version`, or — once CI uses uv (UVP-032) — let `setup-uv` pick up `.python-version` and run a 3.13+3.14 matrix so both supported minors are tested. Pick one source of truth for the runtime rather than three values.
