# BP-CON-05: Add `# syntax=docker/dockerfile:1` directive to ha-core Dockerfile

**Severity:** medium  |  **Rule:** DOCKER-027 (containers-best-practices)  |  **Area:** `.devcontainer/ha-core/Dockerfile`

## Context
The HA Core Functional Test dev container builds from `.devcontainer/ha-core/Dockerfile`. That file currently starts directly with `FROM ...` and omits the BuildKit syntax directive. Without `# syntax=docker/dockerfile:1` on the first line, BuildKit falls back to whatever Dockerfile frontend the host Docker engine ships with, which on older hosts can be years behind the current stable language version.

The consequence is silent feature degradation. Modern Dockerfile features fail with confusing parse errors (e.g. `unknown flag: --mount`) rather than a clear "frontend too old" message:
- `RUN --mount=type=cache` / `--mount=type=secret` / `--mount=type=ssh`
- `RUN <<EOF` heredoc syntax
- `COPY --link` / `COPY --exclude=`
- `# check=...` lint directives and `docker build --check` (needs >= 1.8)

This matters here because the sibling file `.devcontainer/Dockerfile` already has the directive on line 1 (and uses `SHELL ... -o pipefail`, pinned digests, etc.), so the two dev container Dockerfiles are inconsistent. The ha-core file is the one most likely to grow cache mounts (uv installs, HA core builds) where the missing directive would bite. The fix is a single line with no behavioral downside on modern engines.

## Current state
`.devcontainer/ha-core/Dockerfile:1` — the file begins directly with the `FROM` instruction; there is no syntax directive:

```dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/base:debian

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
```

For contrast, `.devcontainer/Dockerfile:1` already does this correctly:

```dockerfile
# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04@sha256:81380e4c9c14e8a629ff39029639e4b7893e67400246fa7782a0fe7dc193a02a
```

## Proposed fix
Prepend the syntax directive as the very first line of `.devcontainer/ha-core/Dockerfile`. This is a parser comment and must be line 1 (before any other comment or instruction) to take effect.

Before:
```dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/base:debian
```

After:
```dockerfile
# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/vscode/devcontainers/base:debian
```

Use the unpinned `docker/dockerfile:1` (auto-tracks the latest stable v1 frontend), matching the sibling `.devcontainer/Dockerfile`. Do not pin a minor version unless a feature that requires `>= 1.8` (e.g. `docker build --check`) is later adopted in this file.

Scope is limited to adding this one line. Do not touch the `FROM` image reference, the `apt-get` layer, the `uv` COPY, or any other instruction — those are separate findings if any.

## Acceptance criteria
- [ ] `.devcontainer/ha-core/Dockerfile` line 1 is exactly `# syntax=docker/dockerfile:1`.
- [ ] The directive precedes every other line, including comments and the `FROM` instruction.
- [ ] No other lines in the file are modified.
- [ ] The dev container still builds: building from `.devcontainer/ha-core/Dockerfile` (or rebuilding the "HA Core Functional Test" container) succeeds.

## Notes / risks
- Low risk: the directive is a BuildKit parser comment; on engines that already use the v1 frontend it is a no-op, and on older engines it pins the correct frontend.
- Must be the literal first line — placing it after another comment silently disables it (BuildKit only reads the directive from the leading comment block).
- Related rules: DOCKER-029 (`docker build --check` lint) requires frontend `>= 1.8`; if that lint is later enabled for this file, bump the pin to `# syntax=docker/dockerfile:1.8`. The unpinned `:1` is sufficient for now.
- No ordering dependency on other tickets; this is a standalone one-line change.
