# BP-UV-04: Add a `uv audit` CI job to scan uv.lock for known CVEs

> **Status: WON'T DO (permanent).** Closed without action by maintainer decision.

**Severity:** medium  |  **Rule:** UVP-028 (uv-best-practices)  |  **Area:** `.github/workflows/` (CI), `uv.lock`

## Context
This repo is a deployable Home Assistant integration that locks networked,
security-sensitive dependencies in `uv.lock` (769 KB lockfile). `pyproject.toml`
pins `homeassistant` and `aiohttp` as runtime dependencies and `litellm<=1.82.6`
in the dev group — all of which regularly receive CVEs (aiohttp request-smuggling
advisories, litellm SSRF/auth advisories, etc.).

A lockfile is a point-in-time snapshot. Once pinned, the locked versions do not
change until someone explicitly runs `uv lock --upgrade-package`. Without an audit
gate, a freshly disclosed CVE against an already-locked version produces **no CI
signal at all** — the only feedback is a dependency-bot PR or a security incident.

`uv audit` queries the OSV database for every version in `uv.lock` and exits
non-zero on findings, turning "is my lockfile shipping known vulnerabilities" into
a yes/no CI gate. It reads the lockfile directly (no `uv sync` needed), so it is
fast and runs pre-install. Today neither CI workflow performs any vulnerability
scanning.

## Current state
Two workflows exist, neither has an audit step.

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

No `audit.yml` exists, and `uv` is not currently installed in either workflow
(lint uses bare `pip install ruff`, not `uv`).

## Proposed fix
Add a new, dedicated workflow `.github/workflows/audit.yml`. A separate file keeps
the audit independently schedulable (daily cron) and avoids coupling its triggers
to the lint/validate jobs.

Create `.github/workflows/audit.yml`:
```yaml
name: Audit

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 6 * * *"   # daily — surfaces newly-disclosed CVEs against the locked deps

jobs:
  audit:
    name: uv audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      # No sync needed — uv audit reads uv.lock directly.
      - run: uv audit --no-dev
```

`--no-dev` scopes the gate to runtime dependencies (`homeassistant`, `aiohttp`,
`voluptuous`), matching the deployable surface. If the team also wants the dev
group (`litellm`, `pytest`, etc.) scanned, drop `--no-dev` — recommend keeping
`--no-dev` for the blocking gate and, if desired, adding a second non-blocking
informational run for dev deps.

For any accepted-risk finding (a CVE in a code path the integration does not
exercise), centralize the allowlist in `pyproject.toml` rather than scattering
`--ignore` flags in CI:
```toml
[tool.uv.audit]
ignore = [
  "GHSA-xxxx-yyyy-zzzz",   # rationale; reviewed YYYY-MM-DD
]
```

## Acceptance criteria
- [ ] `.github/workflows/audit.yml` exists and defines a job that runs `uv audit --no-dev`.
- [ ] The workflow triggers on push to `main`, on pull requests, and on a daily `schedule` cron.
- [ ] The job sets up uv via `astral-sh/setup-uv` (no `uv sync` step — audit reads `uv.lock` directly).
- [ ] Running `uv audit --no-dev` locally against the current `uv.lock` succeeds (exit 0) OR every reported finding is documented under `[tool.uv.audit] ignore` in `pyproject.toml` with a dated rationale.
- [ ] CI fails (non-zero) when the lockfile contains an un-allowlisted OSV finding.

## Notes / risks
- **First run may already fail.** Before merging, run `uv audit --no-dev` locally
  against the committed `uv.lock`. If it reports real CVEs, those must be resolved
  (`uv lock --upgrade-package <pkg>`) or explicitly allowlisted with rationale —
  do not merge a permanently-red gate. This is expected work, not a blocker for
  the ticket.
- **Action SHA pinning (UVP-030) is intentionally out of scope here.** The
  existing workflows pin actions to floating tags (`actions/checkout@v4`,
  `hassfest@master`), so the snippet above matches current repo style. Hardening
  all actions to commit SHAs is a separate, repo-wide finding — pinning only this
  one file would be inconsistent. Track it under its own ticket.
- **Complements, not replaces, PY-077 (`pip-audit`).** `uv audit` scans the
  lockfile graph; `pip-audit` scans an installed environment. Either is sufficient
  for this finding; `uv audit` is preferred because the repo already uses uv and it
  needs no install step.
- **`uv audit` availability.** It requires a recent uv release. `astral-sh/setup-uv@v6`
  installs a current uv; if a pinned older uv is later introduced, confirm it ships
  the `audit` subcommand.
- **When-NOT-to-apply check:** the rule exempts air-gapped CI (no OSV access) and
  template/scaffolding repos. Neither applies — this is a network-connected GitHub
  Actions setup shipping a real integration. The finding is valid.
