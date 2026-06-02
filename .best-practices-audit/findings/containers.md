# Containers Best-Practices Audit — LLM-Home-Controller

**Files reviewed:** `.devcontainer/Dockerfile`, `.devcontainer/devcontainer.json`,
`.devcontainer/ha-core/Dockerfile`, `.devcontainer/ha-core/devcontainer.json`,
`.devcontainer/ha-core/docker-compose.yml`, plus `.gitignore` and a repo-root
`.dockerignore` existence/coverage check.

The primary `.devcontainer/Dockerfile` is in good shape — it pins the base image by
digest, carries `# syntax=docker/dockerfile:1`, sets `SHELL` with `pipefail`, uses
`--no-install-recommends` + apt cleanup, and copies `uv` from a pinned version+digest
distroless image. The weaknesses cluster in the **`ha-core/` functional-test stack**,
which is noticeably less hardened than the main dev container, and in the **absence of a
`.dockerignore`** — which matters because the `ha-core` compose build uses the entire repo
root (`context: ../..`) as its build context.

**Findings by severity:** 3 high, 4 medium, 2 low.

## Summary table

| Rule ID | Severity | Title | Location |
|---|---|---|---|
| UV-001 / SEC-010 | high | `uv` copied from unpinned `:latest` (no version, no digest) | `.devcontainer/ha-core/Dockerfile:28` |
| DOCKER-002 | high | `ha-core` base image not pinned by digest (mutable `:debian` tag) | `.devcontainer/ha-core/Dockerfile:1` |
| SEC-001 / SEC-015 | high | No `.dockerignore` for the repo-root build context | repo root (used by `.devcontainer/ha-core/docker-compose.yml:4-5`) |
| DOCKER-002 | medium | `ollama/ollama` image fully unpinned (no tag, no digest) | `.devcontainer/ha-core/docker-compose.yml:21` |
| DOCKER-027 | medium | `ha-core/Dockerfile` missing `# syntax=docker/dockerfile:1` | `.devcontainer/ha-core/Dockerfile:1` |
| DEVC-018 | medium | `ha-core` service has no `init: true` (no PID-1 reaping) | `.devcontainer/ha-core/docker-compose.yml:2` |
| DEVC-005 | medium | `ha-core` stack mounts no uv cache volume | `.devcontainer/ha-core/docker-compose.yml:6-10` |
| DEVC-009 | low | `ha-core` devcontainer assumes `even-better-toml`/TOML tooling but doesn't declare it | `.devcontainer/ha-core/devcontainer.json:27-34` |
| COMPOSE-009 | low | Long-lived `ollama` service has no `restart` policy | `.devcontainer/ha-core/docker-compose.yml:20-25` |

---

## UV-001 / SEC-010 — `uv` copied from unpinned `:latest`

**Severity:** high · **Location:** `.devcontainer/ha-core/Dockerfile:28`

```dockerfile
# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
```

`:latest` is a mutable tag with no digest. Every rebuild can pull a different `uv` binary,
silently changing dependency-resolution behavior across machines and over time, and there
is no audit trail of which `uv` actually built the environment. The main
`.devcontainer/Dockerfile:17` already does this correctly
(`ghcr.io/astral-sh/uv:0.11.18@sha256:78bc42...`), so the two files have drifted.

**Recommendation:** Pin the literal version+digest pair, matching the main Dockerfile:
`COPY --from=ghcr.io/astral-sh/uv:0.11.18@sha256:78bc42400d77b0678ba95765305c826652ed5431f399257271dda681d0318f03 /uv /usr/local/bin/uv`.

---

## DOCKER-002 — `ha-core` base image not pinned by digest

**Severity:** high · **Location:** `.devcontainer/ha-core/Dockerfile:1`

```dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/base:debian
```

The tag `:debian` is mutable and floating (it also points at the deprecated
`vscode/devcontainers` image path rather than the current `devcontainers/base`). The base
OS, its CVE surface, and its preinstalled toolchain can change underneath the build with no
diff — breaking reproducibility of the functional-test environment. The main Dockerfile pins
its base by digest (`...base:ubuntu-22.04@sha256:81380e...`); this one does not.

**Recommendation:** Pin a specific tag + digest, e.g.
`FROM mcr.microsoft.com/devcontainers/base:debian@sha256:<digest>`, and let Renovate/Dependabot
bump the version+digest as a literal pair (SEC-010).

---

## SEC-001 / SEC-015 — No `.dockerignore` for the repo-root build context

**Severity:** high · **Location:** repo root; consumed by `.devcontainer/ha-core/docker-compose.yml:4-5`

```yaml
build:
  context: ../..
  dockerfile: .devcontainer/ha-core/Dockerfile
```

There is no `.dockerignore` anywhere in the repo. The `ha-core` compose build sets its
context to the repository root (`../..`), so `docker build` sends the **entire working tree**
to the daemon — including a 756 MB `.venv/`, the full `.git/` history (3.2 MB), `llm_home_controller.egg-info/`,
`.ruff_cache/`, `.pytest_cache/`, and any `.env` a developer creates locally. Even though
the `ha-core/Dockerfile` doesn't `COPY . .` today, the oversized context slows every build,
busts the cache on unrelated host-side changes, and is one stray `COPY` away from leaking
`.git/` (SEC-013) or secrets (SEC-012) into a layer.

**Recommendation:** Add a comprehensive `.dockerignore` at the repo root covering the SEC-015
baseline: `.git`, `.github`, `.venv`/`venv`, `__pycache__`, `*.py[cod]`, `.pytest_cache`,
`.mypy_cache`, `.ruff_cache`, `*.egg-info`, `.env`, `.env.*` (with `!.env.example`), `**/*.pem`,
`**/*.key`, `.vscode`, `.idea`, `.DS_Store`, and `Dockerfile*`/`compose*.yml`.

---

## DOCKER-002 — `ollama/ollama` image fully unpinned

**Severity:** medium · **Location:** `.devcontainer/ha-core/docker-compose.yml:20-25`

```yaml
  ollama:
    image: ollama/ollama
```

No tag and no digest — this resolves to `ollama/ollama:latest` and floats freely. The local
LLM backend used for functional testing can change behavior or break between rebuilds with no
visible cause, undermining "works on my machine" reproducibility of test runs.

**Recommendation:** Pin at least a tag, ideally a digest:
`image: ollama/ollama:0.x.y@sha256:<digest>`.

---

## DOCKER-027 — `ha-core/Dockerfile` missing the syntax directive

**Severity:** medium · **Location:** `.devcontainer/ha-core/Dockerfile:1`

The file begins directly with `FROM ...`; there is no `# syntax=docker/dockerfile:1` first
line. Without it, BuildKit falls back to whatever frontend the engine ships, so
`COPY --from`, cache mounts, heredocs, and `--check` lint can silently degrade on older
hosts. The main `.devcontainer/Dockerfile:1` already has the directive, so this is drift
between the two files.

**Recommendation:** Add `# syntax=docker/dockerfile:1` as the first line.

---

## DEVC-018 — `ha-core` compose service has no `init: true`

**Severity:** medium · **Location:** `.devcontainer/ha-core/docker-compose.yml:2`

The `ha-dev` service (`command: sleep infinity`, long-lived dev container running the HA
language server, pytest, debuggers, and forked subprocesses) sets no `init:`. Unlike the main
dev container — which correctly sets `"init": true` in `.devcontainer/devcontainer.json:8` —
the compose-mode container gets no init process, so zombie `<defunct>` children accumulate
over the container's lifetime and SIGTERM handling is unreliable. In `dockerComposeFile` mode
the knob must live on the service, not in `devcontainer.json`.

**Recommendation:** Add `init: true` to the `ha-dev` service.

---

## DEVC-005 — `ha-core` stack mounts no uv cache volume

**Severity:** medium · **Location:** `.devcontainer/ha-core/docker-compose.yml:6-10`

```yaml
    volumes:
      - ../..:/workspace/LLM-Home-Controller:cached
      - ha-core:/workspace/ha-core
      - ha-config:/workspace/ha-config
      - ha-venv:/home/vscode/.local/ha-venv
```

`postCreateCommand: bash scripts/setup-ha-core.sh` clones and `uv pip install -e`'s all of
Home Assistant Core (a multi-minute install). There is no volume mounted at the uv cache
(`~/.cache/uv`), so every container rebuild re-downloads the full HA dependency tree from
scratch. The main dev container already mounts `llmhc-uv-cache` for exactly this reason
(`.devcontainer/devcontainer.json:26`).

**Recommendation:** Add a `uv-cache:/home/vscode/.cache/uv` named volume to the `ha-dev`
service (and declare it under top-level `volumes:`).

---

## DEVC-009 — TOML tooling assumed but not declared in `ha-core` devcontainer

**Severity:** low · **Location:** `.devcontainer/ha-core/devcontainer.json:27-34`

The project edits `pyproject.toml`/`hacs.json` and the main dev container is consistent about
its extension set, but the `ha-core` devcontainer's `extensions` list omits a TOML extension
(e.g. `tamasfe.even-better-toml`) that the main one also omits — minor, but worth aligning so
both surfaces give the same editing experience. This is polish, not correctness.

**Recommendation:** If TOML editing support is desired, add `tamasfe.even-better-toml` to both
devcontainers' `customizations.vscode.extensions`. Otherwise no action needed.

---

## COMPOSE-009 — `ollama` service has no restart policy

**Severity:** low · **Location:** `.devcontainer/ha-core/docker-compose.yml:20-25`

```yaml
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
```

The long-lived `ollama` backend has no `restart:` policy, so a crash (or a Docker daemon
restart) leaves it down and the functional-test environment silently loses its LLM endpoint.
Low severity because this is a local dev-only stack.

**Recommendation:** Add `restart: unless-stopped` to the `ollama` service (the spec's preferred
policy for long-lived services that should honor an explicit stop).

---

### Rules explicitly checked and found clean / not applicable

- **SEC-019 (`.env` in `.gitignore`, not tracked):** `.env` and `.env.local` are gitignored
  (`.gitignore:20-21`); `git ls-files` shows no tracked `.env`; no `.env` file exists. Clean.
- **DEVC-012 (cross-platform mount path):** main devcontainer correctly uses
  `${localEnv:HOME}${localEnv:USERPROFILE}` (`devcontainer.json:21`). Clean.
- **DEVC-015 (agentic CLI state in named volume):** `~/.claude` is a per-worktree named volume
  (`devcontainer.json:24`), not a host bind mount. Clean.
- **DOCKER-019/020 (apt update+install pairing, `--no-install-recommends`):** both Dockerfiles
  pair update+install in one `RUN` and pass `--no-install-recommends`. Clean.
- **DOCKER-005/COMPOSE-023 (non-root / no-new-privileges):** intentionally out of scope — these
  are dev containers, where interactive `sudo` (used by `onCreateCommand`) and a `vscode` user
  are expected; the rules' "When NOT to apply" exempts dev containers.
- **DEVC-006 (`~/.ssh` bind mount):** mounted read-only (`devcontainer.json:21`) and documented
  inline alongside the trusted-container note; acceptable given the documented posture.
