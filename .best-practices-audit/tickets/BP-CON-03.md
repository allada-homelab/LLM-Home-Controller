# BP-CON-03: Add a repo-root `.dockerignore` to scope the ha-core build context

> **Status: DONE** — shipped in commit `0582f6e`. Repo-root `.dockerignore` added; `docker build --check` confirms it is loaded and the context excludes `.git`/`.venv`.

**Severity:** high  |  **Rule:** SEC-001 (containers-best-practices)  |  **Area:** repo root / `.devcontainer/ha-core/` build context

## Context
The HA Core functional-test container builds with the **repository root** as its
build context, but the repo has **no `.dockerignore` anywhere**. Every `docker build`
of that service therefore ships the entire 762 MB working tree to the build daemon —
dominated by a 756 MB `.venv/` — plus `.git/` (full history), `llm_home_controller.egg-info`,
`.ruff_cache`, `.pytest_cache`, `.remember/`, and a 769 KB `uv.lock`.

Why this matters for *this* repo:

1. **Performance / cache.** The 756 MB `.venv/` and the volatile caches (`.pytest_cache`,
   `.ruff_cache`) are uploaded on every build and bust the context hash whenever a test or
   lint run touches them — even though the Dockerfile never copies them. Builds are slow and
   cache hits are rare.
2. **Secrets defense-in-depth (SEC-013).** `.git/` ships the *entire* commit history into
   the build context. `.git/config` can carry an embedded access token from an HTTPS clone;
   rotated/"deleted" secrets remain recoverable from packfiles. No `.env`/`*.pem`/`*.key`
   files exist in the root today, so the `.env` exclusions are preventative — but `.git/`
   exposure is live now.

The Dockerfiles in this repo do not use a blanket `COPY . .`
(`.devcontainer/ha-core/Dockerfile` copies via `COPY --from=...` and dev-container tooling),
so nothing leaks into the *final image* today. The concrete harm is the bloated/leaky
**build context** itself. A `.dockerignore` is the standard fix and SEC-001 says every
project should have one.

## Current state
`/workspace/LLM-Home-Controller/.devcontainer/ha-core/docker-compose.yml:1-6` — context is the repo root:

```yaml
services:
  ha-dev:
    build:
      context: ../..                                  # = repo root (762 MB)
      dockerfile: .devcontainer/ha-core/Dockerfile
```

Repo root listing (abridged) — no `.dockerignore`, large/volatile dirs present:

```
.venv/                         756M
.git/                          3.2M
llm_home_controller.egg-info/   28K
.ruff_cache/  .pytest_cache/  .remember/  .vscode/  .idea(n/a)
research.md (58K)  uv.lock (769K)
```

`find . -name .dockerignore` returns nothing.

There is an existing `/workspace/LLM-Home-Controller/.gitignore` whose ignore set
overlaps heavily (`.venv/`, `__pycache__/`, `*.egg-info/`, the caches, `.env`).
Note `.gitignore` is **not** consulted by `docker build` — a separate `.dockerignore`
is required.

## Proposed fix
Add a single new file `/workspace/LLM-Home-Controller/.dockerignore` (build contexts use
the `.dockerignore` adjacent to the context root; for `context: ../..` that is the repo root).
Surgical, additive — no existing file changes.

```
# === .dockerignore (repo root) ===
# VCS (SEC-013: .git ships full history + possible tokens in .git/config)
.git
.gitignore
.gitattributes
.github

# Python
__pycache__
*.py[cod]
*$py.class
.venv
venv
.pytest_cache
.mypy_cache
.ruff_cache
*.egg-info
build
dist

# Environment / secrets (defense in depth)
.env
.env.*
!.env.example
**/*.pem
**/*.key

# Editors / OS
.vscode
.idea
.DS_Store

# Docker itself
Dockerfile*
.dockerignore
compose*.yml
docker-compose*.yml
```

This matches the SEC-001 / SEC-015 baseline in the recommendation. The
`!.env.example` negation deliberately re-includes the template while excluding real
`.env*` files.

## Acceptance criteria
- [ ] `/workspace/LLM-Home-Controller/.dockerignore` exists at the repo root.
- [ ] It excludes (at minimum): `.git`, `.github`, `.venv`/`venv`, `__pycache__`,
      `*.py[cod]`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `*.egg-info`,
      `.env`, `.env.*` (with `!.env.example`), `**/*.pem`, `**/*.key`, `.vscode`,
      `.idea`, `.DS_Store`, `Dockerfile*`, `compose*.yml`/`docker-compose*.yml`.
- [ ] Build context size drops dramatically: building `ha-dev` no longer uploads
      `.venv/` or `.git/`. Verify with e.g.
      `docker buildx build -f .devcontainer/ha-core/Dockerfile .` and confirm the
      "transferring context" size is single-digit MB, not ~760 MB
      (or inspect via `docker buildx build --no-cache` build logs).
- [ ] `ha-dev` still builds successfully and HA still starts (the integration is
      mounted via the `../..:/workspace/LLM-Home-Controller` bind, not copied at
      build time, so excluding source from the *context* does not break runtime).

## Notes / risks
- **Bind-mount, not COPY.** `ha-core/docker-compose.yml` mounts the working copy as a
  volume at runtime; the build context is used only by the Dockerfile's `COPY`/`RUN`
  steps. Excluding source files from the context is therefore safe here. If a future
  Dockerfile edit adds `COPY src/ ...`, make sure the needed paths are *not* matched by
  an exclude.
- **Root `.devcontainer/devcontainer.json` is unaffected.** That build uses the default
  context (`.devcontainer/`), so this root `.dockerignore` does not apply to it. If you
  later want the same hygiene there, a separate `.devcontainer/.dockerignore` would be
  needed — out of scope for this ticket.
- **Related rules:** SEC-013 (`.git` exclusion — covered by the first block),
  SEC-015 (no real secrets in the context — the `.env*`/`*.pem`/`*.key` lines).
- **Do not delete `.gitignore`** — it serves a different purpose; this ticket only *adds*
  `.dockerignore`.
