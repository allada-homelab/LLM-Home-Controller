# BP-CON-09: Add `restart: unless-stopped` to the ollama service

**Severity:** low  |  **Rule:** COMPOSE-009 (containers-best-practices)  |  **Area:** `.devcontainer/ha-core/docker-compose.yml` — ollama dev-test backend service

## Context
The `ollama` service is the long-lived OpenAI-compatible LLM backend used for functional testing of the integration inside the HA Core dev container. It is meant to stay running for the duration of a dev session so the `ha-dev` container can reach it at `http://ollama:11434`.

It currently declares no `restart:` policy, so it inherits Compose's default of `no`. If the ollama process crashes, or the Docker daemon restarts (e.g. host reboot, Docker Desktop update), the container stays exited and the integration's functional tests silently lose their backend. A developer then has to notice the failure and manually `docker compose up` the service again.

Per COMPOSE-009, long-lived services that "should be running" want `restart: unless-stopped`: it recovers from crashes and daemon restarts, but still honors an explicit `docker compose stop` so a developer who deliberately stops ollama isn't fought by Docker.

## Current state
`.devcontainer/ha-core/docker-compose.yml:20-25`

```yaml
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
```

No `restart:` key is present on the service.

## Proposed fix
Add a single `restart: unless-stopped` line to the `ollama` service. Minimal, surgical, no other changes.

Before:
```yaml
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
```

After:
```yaml
  ollama:
    image: ollama/ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
```

## Acceptance criteria
- [ ] The `ollama` service in `.devcontainer/ha-core/docker-compose.yml` declares `restart: unless-stopped`.
- [ ] No other service or value in the file is modified.
- [ ] `docker compose -f .devcontainer/ha-core/docker-compose.yml config` parses without error (valid YAML / valid Compose schema).

## Notes / risks
- Scope is the `ollama` service only, per this finding. The `ha-dev` service intentionally runs `command: sleep infinity` (standard dev-container idle pattern); it is started/stopped by the dev-container tooling rather than expected to self-recover, so a restart policy there is a separate judgement call and is out of scope for this ticket.
- Related: COMPOSE-018 notes that long-lived services in COMPOSE-009 territory should also declare an explicit `logging:` block (e.g. json-file with rotation). That is a distinct finding — do not bundle it here unless a separate ticket calls for it.
- No tests reference this compose file's restart behavior, so there is no test breakage risk. The change only affects local/dev-container runtime resilience, not CI or the shipped integration.
