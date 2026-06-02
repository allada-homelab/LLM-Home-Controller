# BP-UV-01: Validate the committed uv.lock in CI via a uv-based sync job

> **Status: DONE** — shipped in commit `0582f6e`. `lint.yml` rewritten to a uv job (`uv lock --check` + `uv sync --locked --dev` + ruff + pytest), actions SHA-pinned. Required adding `PyTurboJPEG` to the dev group so the suite collects.

**Severity:** high  |  **Rule:** UVP-032 (uv-best-practices)  |  **Area:** `.github/workflows/lint.yml`, `.github/workflows/validate.yml`

## Context
This repo commits a `uv.lock` (~750 KB) and declares its dev tooling in `[dependency-groups]` of `pyproject.toml` (pytest, pytest-asyncio, pytest-homeassistant-custom-component, ruff, hassil, etc.). Despite that, no CI workflow ever exercises uv. Consequently:

1. **The lockfile is never validated.** A developer can add or bump a dependency in `pyproject.toml`, forget to run `uv lock`, and commit a stale `uv.lock`. CI stays green because nothing runs `uv sync --locked` or `uv lock --check`. The first person to discover the drift is the next developer whose local `uv sync --locked` fails with "lockfile is out of date" — or production, if a deploy pipeline relocks silently. This is the exact failure mode UVP-032 exists to catch (uv#12372).
2. **Lint does not run the locked toolchain.** `lint.yml` installs ruff via bare `pip install ruff`, which fetches whatever the latest ruff is at CI time — not the `ruff>=0.9` version captured in `uv.lock`. A ruff release that changes lint/format behavior can break CI (or, worse, pass CI but disagree with what a developer sees locally) with no source change. The locked deps are never installed, so the tests never run against the environment the lockfile describes. (`validate.yml` only runs hassfest and touches no Python deps at all.)

Net effect: the lockfile and dependency groups are committed but unenforced — they provide a false sense of reproducibility.

## Current state

`.github/workflows/lint.yml` (full file):
```yaml
name: Lint

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

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

`.github/workflows/validate.yml` (full file):
```yaml
name: Validate

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  hassfest:
    name: Hassfest
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: home-assistant/actions/hassfest@master
```

Relevant `pyproject.toml` facts: `requires-python = ">=3.13"`, dev tooling under `[dependency-groups].dev` (includes `ruff>=0.9`, `pytest>=8.0`). `uv.lock` is present at repo root. A `.python-version` file exists and pins `3.14` — note this disagrees with the hardcoded `python-version: "3.13"` in `lint.yml` (see Notes).

## Proposed fix
Replace the pip-based ruff job in `lint.yml` with a uv-based job that validates the lockfile and runs ruff + pytest through the locked environment. This is the minimal change that satisfies UVP-032 (every `uv sync` carries `--locked`) and folds in the existing lint plus the currently-unrun test suite.

Rewrite `.github/workflows/lint.yml`:
```yaml
name: Lint

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  uv:
    name: uv (lockfile + ruff + pytest)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<sha>          # v5.x — see UVP-030
      - uses: astral-sh/setup-uv@<sha>        # v8.x — see UVP-030
        with:
          enable-cache: true
      - name: Verify lockfile is current
        run: uv lock --check
      - run: uv sync --locked --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest tests/
```

`setup-uv` reads `.python-version` automatically, so no `python-version:` input is needed (the interpreter comes from the pinned file rather than a hardcoded string — this also removes the 3.13-vs-3.14 mismatch). `enable-cache: true` caches `~/.cache/uv` keyed on `uv.lock` (UVP-031); do not add `actions/cache` on top of it and do not cache `.venv/`.

Leave `validate.yml` (hassfest) unchanged — it is orthogonal to the lockfile and out of scope for this finding.

Pin both actions to a commit SHA with the version tag in a trailing comment (UVP-030). To resolve the SHAs:
```bash
gh api repos/astral-sh/setup-uv/git/refs/tags/v8.1.0 --jq '.object.sha'
gh api repos/actions/checkout/git/refs/tags/v5.0.0  --jq '.object.sha'
```

## Acceptance criteria
- [ ] `lint.yml` runs `uv lock --check` as an explicit step that fails the job if `uv.lock` is out of date relative to `pyproject.toml`.
- [ ] Every `uv sync` in CI carries `--locked` (no bare `uv sync` anywhere in `.github/workflows/`).
- [ ] The dev dependency group is installed via `uv sync --locked --dev`, and ruff is invoked as `uv run ruff ...` (locked version), not `pip install ruff` + `ruff`.
- [ ] The test suite runs in CI via `uv run pytest tests/` against the locked environment (previously never run in CI).
- [ ] `actions/checkout` and `astral-sh/setup-uv` are pinned to commit SHAs with a `# vX.Y.Z` comment (UVP-030).
- [ ] No `actions/setup-python` step and no `pip install` remain in `lint.yml`.
- [ ] Verified by intentionally desyncing the lockfile (e.g. bump a constraint in `pyproject.toml` without relocking) on a scratch branch and confirming the `uv lock --check` step fails; revert before merge.

## Notes / risks
- **Python-version mismatch (adjacent, in-scope to flag):** `.python-version` pins `3.14` while `lint.yml` currently hardcodes `3.13` and `pyproject.toml` only sets `requires-python = ">=3.13"`. Letting `setup-uv` consume `.python-version` resolves the CI side to 3.14. Confirm the project actually intends to target 3.14 in CI; if 3.13 is intended, fix `.python-version` rather than re-hardcoding a version in the workflow. Do not silently change `pyproject.toml`'s `requires-python` as part of this ticket.
- **Test runtime / network:** `pytest-homeassistant-custom-component` and the HA core deps are heavy. The cached uv environment mitigates cold-install cost, but the first run will be slow and downloads HA wheels. If the suite needs services or network the existing local runs assume, scope that separately — this ticket only adds the CI invocation.
- **Related rules:** UVP-030 (SHA-pin actions) and UVP-031 (let `setup-uv` cache, never cache `.venv`) are folded into the fix above. UVP-028 (`uv audit` in CI) is a separate, complementary gate — out of scope here; track separately if desired.
- **Ordering:** No dependency on other audit tickets. Container-side uv rules (UV-00x) do not apply — this repo's CI is not container-built.
- This is a real gap, not a non-issue: the lockfile and dependency groups exist but are wholly unenforced by CI.
