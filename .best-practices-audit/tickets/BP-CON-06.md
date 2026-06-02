# BP-CON-06: Add `init: true` to the ha-dev compose service for PID-1 zombie reaping

> **Status: DONE** — shipped in commit `0582f6e`. `init: true` added to `ha-dev`; `docker compose config` shows it in the resolved config.

**Severity:** medium  |  **Rule:** DEVC-018 (containers-best-practices)  |  **Area:** `.devcontainer/ha-core/docker-compose.yml` (HA Core Functional Test dev container)

## Context
The "HA Core Functional Test" dev container is started in `dockerComposeFile` mode (`.devcontainer/ha-core/devcontainer.json` sets `"dockerComposeFile": "docker-compose.yml"` and `"service": "ha-dev"`). The `ha-dev` service runs `command: sleep infinity`, so `sleep` becomes PID 1 inside the container.

`sleep` is not an init process: it does not reap orphaned/zombie child processes. The container is a long-lived development environment where editors, language servers (Pylance, debugpy), the Home Assistant process, pytest runs, and `git` repeatedly fork children. Over the lifetime of the container those orphaned processes are reparented to PID 1 and, with no reaper, accumulate as `<defunct>` entries in the process table. This is exactly the scenario DEVC-018 targets: a slow/messy container stop (the daemon waits for a SIGTERM response PID 1 never handles) and a process table filling with zombies after extended editing sessions.

The image itself does not run a real init. `.devcontainer/ha-core/Dockerfile` (`FROM mcr.microsoft.com/vscode/devcontainers/base:debian`) defines no `ENTRYPOINT` and installs no `tini`/`dumb-init`/`s6-overlay`, so the "When NOT to apply" exemption in DEVC-018 does not hold here.

In `dockerComposeFile` mode the dev container `"init"`/`"runArgs": ["--init"]` fields do not apply; the fix must be set on the compose service via `init: true` (Docker adds a tini-based init as PID 1, which then runs `sleep infinity` as a child and reaps zombies).

## Current state
`.devcontainer/ha-core/docker-compose.yml:2-15`:

```yaml
services:
  ha-dev:
    build:
      context: ../..
      dockerfile: .devcontainer/ha-core/Dockerfile
    volumes:
      - ../..:/workspace/LLM-Home-Controller:cached
      - ha-core:/workspace/ha-core
      - ha-config:/workspace/ha-config
      - ha-venv:/home/vscode/.local/ha-venv
    ports:
      - "8123:8123"
    environment:
      - PYTHONASYNCIODEBUG=1
    command: sleep infinity
```

There is no `init:` key on the `ha-dev` service.

For reference, `.devcontainer/ha-core/devcontainer.json:3-4` confirms compose mode and the targeted service:

```jsonc
"dockerComposeFile": "docker-compose.yml",
"service": "ha-dev",
```

## Proposed fix
Add `init: true` to the `ha-dev` service in `.devcontainer/ha-core/docker-compose.yml`. Minimal, single-line change; no need to touch `devcontainer.json` or the Dockerfile.

Before:

```yaml
  ha-dev:
    build:
      context: ../..
      dockerfile: .devcontainer/ha-core/Dockerfile
    volumes:
```

After:

```yaml
  ha-dev:
    build:
      context: ../..
      dockerfile: .devcontainer/ha-core/Dockerfile
    init: true
    volumes:
```

(Placement within the service block is not significant; placing it near the top keeps it visible.)

The `ollama` service is out of scope for this ticket — it runs an external image and is not the dev container target. Consider it only if a separate finding covers it.

## Acceptance criteria
- [ ] `.devcontainer/ha-core/docker-compose.yml` has `init: true` on the `ha-dev` service.
- [ ] The compose file still parses: `docker compose -f .devcontainer/ha-core/docker-compose.yml config` succeeds.
- [ ] Inside a rebuilt dev container, PID 1 is a tini-style init rather than `sleep` — e.g. `docker inspect <container> --format '{{.HostConfig.Init}}'` returns `true`, or `ps -p 1 -o comm=` does not return `sleep`.
- [ ] No change required to `devcontainer.json` or the `Dockerfile`.

## Notes / risks
- Related rule: DOCKER-006 (PID 1 / signal handling for runtime images) — same rationale applied to dev containers. DEVC-018 is the dev-container-specific variant.
- Low risk: `init: true` only inserts an init process as PID 1; `sleep infinity` continues to run as its child. No behavior change to HA, ports, or volumes.
- Requires a dev container rebuild to take effect (the `--init` flag is applied at container creation, not on restart).
- No tests reference this compose file, so nothing in `tests/` breaks.
