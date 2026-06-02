# BP-CON-07: Mount a uv cache volume for the ha-core dev stack

> **Status: DONE** — `uv-cache:/home/vscode/.cache/uv` mounted on `ha-dev` + declared top-level; `setup-ha-core.sh` chown extended to `/home/vscode/.cache`. `docker compose config` validates.

**Severity:** medium  |  **Rule:** DEVC-005 (containers-best-practices)  |  **Area:** `.devcontainer/ha-core/docker-compose.yml` (HA Core Functional Test dev container)

## Context
The repo ships two dev container configurations:

1. `.devcontainer/devcontainer.json` — the default container (`uv sync` of this project). It already mounts a persistent uv download cache: `source=llmhc-uv-cache,target=/home/vscode/.cache/uv,type=volume` (line 26).
2. `.devcontainer/ha-core/` — the "HA Core Functional Test" container. It builds from `docker-compose.yml`, and its `postCreateCommand` runs `scripts/setup-ha-core.sh`, which does a full `uv pip install -e . -r requirements_test.txt colorlog` of Home Assistant Core (`setup-ha-core.sh:44-48`). HA Core has a very large dependency tree, so this is the single most expensive step in the setup (the script itself warns "this may take several minutes on first run").

The ha-core compose stack mounts named volumes for the workspace, `ha-core`, `ha-config`, and the venv (`ha-venv`), but nothing at `~/.cache/uv`. Because `uv`'s download/build cache lives on the container's ephemeral root filesystem, every "Rebuild Container" wipes it and the next setup re-downloads and re-builds the entire HA Core dependency set from scratch. This is exactly the waste DEVC-005 targets, and it is inconsistent with the sibling default container which already does the right thing.

Note the venv is on a named volume (`ha-venv`), so an already-installed venv survives rebuilds. But any operation that actually populates the venv — first-time setup, `git pull` + reinstall of HA Core (documented in `CLAUDE.md` and `setup-ha-core.sh:109`), or a `rm -rf` reset — hits the cache, and that cache is gone after a rebuild.

## Current state
`.devcontainer/ha-core/docker-compose.yml:6-10` (service volumes) and `:27-31` (top-level volumes):

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
    ...

volumes:
  ha-core:
  ha-config:
  ha-venv:
  ollama-data:
```

There is no `UV_CACHE_DIR` override or `--no-cache` flag anywhere in `.devcontainer/` or `scripts/`, so uv uses its default cache at `/home/vscode/.cache/uv` — which is unmounted.

Related: `scripts/setup-ha-core.sh:15-17` already chowns the named-volume mount points to the `vscode` user (named volumes mount root-owned — DEVC-010), but the list is `"$HA_CORE_DIR" "$HA_CONFIG_DIR" "$VENV_DIR" /home/vscode/.local`. It does **not** include `/home/vscode/.cache`, so a newly added cache volume would mount root-owned and `uv` (running as `vscode`) would fail to write to it.

## Proposed fix
Two small, surgical edits.

1. `.devcontainer/ha-core/docker-compose.yml` — add the cache volume to the `ha-dev` service and declare it at the top level. Use a name consistent with the existing stack (`uv-cache`); the default container uses `llmhc-uv-cache`, but compose prefixes project names onto its own volumes, so a plain `uv-cache` key is fine and matches the existing `ha-core`/`ha-config` style.

   Before (service volumes):
   ```yaml
       volumes:
         - ../..:/workspace/LLM-Home-Controller:cached
         - ha-core:/workspace/ha-core
         - ha-config:/workspace/ha-config
         - ha-venv:/home/vscode/.local/ha-venv
   ```
   After:
   ```yaml
       volumes:
         - ../..:/workspace/LLM-Home-Controller:cached
         - ha-core:/workspace/ha-core
         - ha-config:/workspace/ha-config
         - ha-venv:/home/vscode/.local/ha-venv
         - uv-cache:/home/vscode/.cache/uv
   ```

   Before (top-level volumes):
   ```yaml
   volumes:
     ha-core:
     ha-config:
     ha-venv:
     ollama-data:
   ```
   After:
   ```yaml
   volumes:
     ha-core:
     ha-config:
     ha-venv:
     ollama-data:
     uv-cache:
   ```

2. `scripts/setup-ha-core.sh:17` — extend the existing ownership-fix line so the new cache mount is owned by `vscode` (it otherwise mounts root-owned and breaks uv writes).

   Before:
   ```bash
   sudo chown -R "$(id -u):$(id -g)" "$HA_CORE_DIR" "$HA_CONFIG_DIR" "$VENV_DIR" /home/vscode/.local 2>/dev/null || true
   ```
   After:
   ```bash
   sudo chown -R "$(id -u):$(id -g)" "$HA_CORE_DIR" "$HA_CONFIG_DIR" "$VENV_DIR" /home/vscode/.local /home/vscode/.cache 2>/dev/null || true
   ```

No other files need to change. Do not touch the default `.devcontainer/devcontainer.json` — it already complies.

## Acceptance criteria
- [ ] `.devcontainer/ha-core/docker-compose.yml` mounts `uv-cache:/home/vscode/.cache/uv` on the `ha-dev` service.
- [ ] `uv-cache` is declared under the top-level `volumes:` key in the same file.
- [ ] `scripts/setup-ha-core.sh` chowns `/home/vscode/.cache` (or `/home/vscode/.cache/uv`) to the `vscode` user so uv can write to the freshly mounted volume.
- [ ] `docker compose -f .devcontainer/ha-core/docker-compose.yml config` parses without error (valid YAML + valid compose schema).
- [ ] After a container rebuild, `/home/vscode/.cache/uv` is non-empty following a prior setup run (i.e., the cache survived the rebuild). A practical check: rebuild, then re-run `setup-ha-core.sh` and confirm the HA Core install resolves from cache rather than re-downloading everything.

## Notes / risks
- The ha-core stack uses compose `volumes:`/service-volume syntax, NOT the devcontainer `mounts:` array used by the sibling default container. The DEVC-005 reference example shows the `mounts:` form; the correct translation for a `dockerComposeFile`-based dev container is the compose syntax above. Don't add a `mounts` entry to this container's `devcontainer.json` — it would be the wrong mechanism for a compose-based service.
- Ownership ordering matters: the chown in `setup-ha-core.sh` runs at the very start of `postCreateCommand`, before the first `uv pip install`, so the cache dir will be writable before uv first touches it. Keep the chown in Step 0; do not move it.
- Pairs with DEVC-010 (named volumes mount root-owned) — that is the reason for edit #2.
- Low blast radius: the only behavioral change is cache persistence. If the cache ever gets corrupted, `docker volume rm <project>_uv-cache` clears it; uv re-populates on next install.
- The default container's volume is named `llmhc-uv-cache` and is shared across worktrees via `${devcontainerId}`-free naming; the compose volume here is project-scoped and separate. That is acceptable — the two containers run different Python environments (this-project venv vs HA Core venv) and need not share a cache. Do not attempt to unify them with an `external:` volume unless explicitly requested; that adds setup friction (the volume must be pre-created) for little gain.
