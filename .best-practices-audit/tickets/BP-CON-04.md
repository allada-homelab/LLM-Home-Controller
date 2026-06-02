# BP-CON-04: Pin the `ollama/ollama` image to a tag (ideally a digest)

**Severity:** medium  |  **Rule:** DOCKER-002 (containers-best-practices)  |  **Area:** `.devcontainer/ha-core/docker-compose.yml` — `ollama` functional-test backend service

## Context
The dev container's `ollama` service references the image as bare `ollama/ollama`, with
no tag and no digest. Docker resolves a bare image reference to the `:latest` tag, which
is mutable: `ollama/ollama:latest` today is not guaranteed to be the same blob next week.

This is the OpenAI-compatible LLM backend developers spin up to functionally test the
integration (see `CLAUDE.md` → "Functional Testing"). Because the tag floats, two
developers (or the same developer at two points in time) can silently end up running
different Ollama versions. Ollama's API surface and model-compatibility have changed
across releases, so a floating image can produce non-reproducible behavior, confuse
"works on my machine" debugging of the conversation agent, and obscure supply-chain
swaps — all of which DOCKER-002 is meant to prevent.

Scope note: this is a developer-tooling/functional-test container, not a shipped
production artifact (this repo is a Home Assistant custom integration; the integration
itself is plain Python and ships no image). So the blast radius is limited to local dev
reproducibility — which is why this is medium, not high — but the fix is cheap and
worth doing.

The same file also has an unpinned-but-built `ha-dev` service (`build:` block, lines
2-15); that one builds from a local Dockerfile and is out of scope for this image-pinning
finding. The base image inside `.devcontainer/ha-core/Dockerfile` is a separate concern
tracked under its own finding, not here.

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

Line 21 (`image: ollama/ollama`) is the offending reference.

## Proposed fix
Pin the image. Minimal acceptable change is a tag; the rule's stated ideal is a tag
plus a digest so the pull is bit-for-bit reproducible.

Before:

```yaml
    image: ollama/ollama
```

After (preferred — tag + digest, with the human-readable tag kept as a trailing comment
per DOCKER-002's guidance for repos without Renovate/Dependabot):

```yaml
    # ollama/ollama:0.x.y — update digest when bumping the tag
    image: ollama/ollama:0.x.y@sha256:<digest>
```

To obtain a real, current tag + digest before committing (do not invent the digest):

```bash
# pick a concrete released tag from https://hub.docker.com/r/ollama/ollama/tags
docker pull ollama/ollama:0.x.y
docker inspect --format '{{index .RepoDigests 0}}' ollama/ollama:0.x.y
# -> ollama/ollama@sha256:<digest>   (combine with the tag in the compose file)
```

If digest pinning is deemed too high-maintenance for a dev-only service, a tag alone
(`image: ollama/ollama:0.x.y`) still satisfies the minimum bar of this finding and
removes the `:latest` float.

Keep the change scoped to line 21 only — do not touch the `ha-dev` build block, ports,
volumes, or the `volumes:` section.

## Acceptance criteria
- [ ] `.devcontainer/ha-core/docker-compose.yml` no longer references `ollama/ollama`
      without a tag; the `image:` value includes at least a version tag.
- [ ] Ideally the reference is `ollama/ollama:<tag>@sha256:<digest>` with the digest
      obtained from an actual pulled image (not fabricated).
- [ ] The pinned tag is a real, published `ollama/ollama` tag from Docker Hub.
- [ ] `docker compose -f .devcontainer/ha-core/docker-compose.yml config` parses without
      error and the `ollama` service still maps port `11434:11434` with the
      `ollama-data` volume mounted at `/root/.ollama`.
- [ ] No unrelated lines in the file changed.

## Notes / risks
- Verify the chosen tag exists for the linux platform the dev container targets; some
  Ollama tags differ by architecture, though `ollama/ollama` is multi-arch.
- A digest pin means devs must bump the digest to get Ollama updates. Mitigate by leaving
  the tag in a comment (as shown) and/or wiring Renovate/Dependabot later — out of scope here.
- Related findings in the same file: COMPOSE-009 (add `restart:` policy to this same
  `ollama` service). These edits are adjacent but independent; either order is fine, just
  avoid stepping on each other's diff in lines 20-25.
- Do not change the `ha-dev` service — its image comes from a local `build:`, not a
  registry tag, so DOCKER-002 image-pinning does not apply to it.
