# BP-UV-02: Install ruff from the lockfile in CI via `uv sync` instead of raw `pip install`

> **Status: DONE** — shipped in commit `0582f6e`, folded into the BP-UV-01 `lint.yml` rewrite (ruff now runs via `uv run` against the locked version).

**Severity:** medium  |  **Rule:** UVP-040 (uv-best-practices)  |  **Area:** `.github/workflows/lint.yml` / CI tooling

## Context
The repo is a uv project: it ships a `uv.lock` (768 KB, committed) and declares `ruff>=0.9` in `[dependency-groups] dev` in `pyproject.toml`. The lockfile currently pins `ruff==0.15.1` (`uv.lock:4015-4020`). Developers running `uv sync --dev` get exactly that version.

The Lint CI workflow does **not** use uv at all. It runs `pip install ruff`, which resolves and installs whatever the latest ruff release is at the moment the job runs — independent of the lockfile. This produces three concrete problems:

1. **CI and local lint can disagree.** When upstream publishes a new ruff (e.g. 0.16.x) that adds a rule or changes formatting, CI starts failing on code that passes locally against the locked 0.15.1 — or vice versa. Lint outcome depends on release timing, not on the committed source.
2. **It bypasses the version the project explicitly pins.** The lockfile exists precisely to make tool versions reproducible; CI ignores it.
3. **It is non-reproducible across reruns.** Re-running an old CI job can produce a different ruff than the original run.

This is the canonical UVP-040 failure: a project-context tool (the linter the team uses) is installed out-of-band rather than from the lockfile, so "the version everyone uses" and "the version CI uses" drift apart.

## Current state
`.github/workflows/lint.yml:1-21` (entire file):

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
      - run: pip install ruff          # <-- line 18: unpinned, bypasses uv.lock
      - run: ruff check .
      - run: ruff format --check .
```

For reference, `pyproject.toml:13-22` already declares the tool:

```toml
[dependency-groups]
dev = [
    ...
    "ruff>=0.9",
]
```

and `uv.lock` pins it (`uv.lock:2238` → `{ name = "ruff", specifier = ">=0.9" }`, resolved to `ruff==0.15.1` at `uv.lock:4015`).

## Proposed fix
Replace the `setup-python` + `pip install ruff` steps with `setup-uv` + a locked sync, then invoke ruff through `uv run` so CI uses the lockfile-pinned version.

**After** (`.github/workflows/lint.yml`):

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
      - uses: astral-sh/setup-uv@<sha> # vX.Y.Z  (pin to a commit SHA — see UVP-030)
        with:
          enable-cache: true
      - run: uv sync --locked --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
```

Scoped notes for the implementer:
- The `actions/setup-python` step can be dropped — `uv sync` provisions the interpreter (the project requires `>=3.13`; `.python-version` / `requires-python` drive selection). Keep it only if a project policy requires it.
- `--locked` makes the job fail loudly if `uv.lock` is stale relative to `pyproject.toml` (UVP-032), which is the desired CI behavior.
- This change touches only `lint.yml`. Do not modify `validate.yml` (hassfest) or `pyproject.toml` — ruff is already correctly declared there.

## Acceptance criteria
- [ ] `.github/workflows/lint.yml` no longer contains `pip install ruff` (or any raw `pip install`).
- [ ] The workflow runs `uv sync --locked --dev` before invoking ruff.
- [ ] ruff is invoked via `uv run ruff check .` and `uv run ruff format --check .`.
- [ ] The ruff version exercised in CI matches the lockfile (`ruff==0.15.1` as currently locked); confirm via the CI log of `uv run ruff --version`, or by adding a temporary `- run: uv run ruff --version` step.
- [ ] CI lint job passes on a clean checkout of `main`.

## Notes / risks
- **Behavior change risk:** CI currently lints against latest ruff; pinning to the locked 0.15.1 may surface or suppress findings relative to today's job. If the locked version flags new issues, fix the code or bump the pin via `uv lock --upgrade-package ruff` (a separate, deliberate change) — do not loosen `--locked`.
- **Related rule UVP-030 (action SHA pinning):** the example pins `setup-uv` to a commit SHA with a version comment. The repo currently uses floating tags (`actions/checkout@v4`, `home-assistant/actions/hassfest@master`). Pinning all actions to SHAs is a broader supply-chain hardening task and is **out of scope** for this ticket; at minimum pin `setup-uv` per the snippet. Resolve a SHA with `gh api repos/astral-sh/setup-uv/git/refs/tags/<tag> --jq '.object.sha'`.
- **Related rule UVP-031 (caching):** `enable-cache: true` lets `setup-uv` cache `~/.cache/uv` keyed on `uv.lock`. Do not add a separate `actions/cache` for the venv.
- No ordering dependency on other audit tickets. This is self-contained within `lint.yml`.
