# BP-CON-02: Pin the ha-core dev container base image by tag + digest

> **Status: WON'T DO (permanent).** Closed without action by maintainer decision.

**Severity:** high  |  **Rule:** DOCKER-002 (containers-best-practices)  |  **Area:** `.devcontainer/ha-core/Dockerfile` (HA Core Functional Test dev container)

## Context
The "HA Core Functional Test" dev container builds from `.devcontainer/ha-core/Dockerfile`,
which references its base image by the mutable `:debian` tag only:

```dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/base:debian
```

Tags are mutable. `:debian` resolves to a different blob whenever Microsoft republishes
the image, so two `docker build` runs on different days produce different toolchains with
no audit trail. That breaks reproducibility, hides supply-chain swaps, and can silently
introduce (or fix) CVEs without any diff to review. For a dev container that downloads and
runs Home Assistant Core from source, this is the layer everything else is built on.

This finding is confirmed and real. Notably, the **sibling** `.devcontainer/Dockerfile`
(the default dev container) already does this correctly and is the in-repo template to follow:

```dockerfile
# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04@sha256:81380e4c9c14e8a629ff39029639e4b7893e67400246fa7782a0fe7dc193a02a
```

Two additional defects in the same `FROM` line, both in scope because the fix touches exactly this line:
1. **Deprecated image path.** `mcr.microsoft.com/vscode/devcontainers/base` is the old path.
   The current/canonical path (used by the sibling Dockerfile and by the rule's recommendation)
   is `mcr.microsoft.com/devcontainers/base` (no `vscode/` segment). The old path still works
   today but is the legacy alias.
2. The file is also missing the `# syntax=docker/dockerfile:1` directive that the sibling has
   (rule DOCKER-005) — called out under Notes, not required by this ticket.

There is **no Renovate or Dependabot config** in this repo (no `renovate.json`, no
`.github/dependabot.yml`), so the rule's fallback guidance applies: keep a human-readable
tag comment next to the digest so a reader knows what version the digest corresponds to.

## Current state
`.devcontainer/ha-core/Dockerfile:1`

```dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/base:debian
```

Consumed by `.devcontainer/ha-core/docker-compose.yml:5` (`dockerfile: .devcontainer/ha-core/Dockerfile`),
which in turn backs `.devcontainer/ha-core/devcontainer.json` ("HA Core Functional Test").

## Proposed fix
Replace line 1 with the canonical image path pinned to the current multi-arch index digest,
keeping the tag as a trailing comment for readability (no auto-updater exists in this repo).

Before:
```dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/base:debian
```

After:
```dockerfile
FROM mcr.microsoft.com/devcontainers/base:debian@sha256:a07f1804d8e665ff64bf6adcd60f4d9b9223dc52ee18a1f1a7a8e1ee1a134d34  # debian, pinned 2026-06-02
```

Digest `sha256:a07f1804d8e665ff64bf6adcd60f4d9b9223dc52ee18a1f1a7a8e1ee1a134d34` is the
multi-arch OCI image index for `mcr.microsoft.com/devcontainers/base:debian` as resolved on
2026-06-02 via `docker buildx imagetools inspect`. Pinning the index digest (not a
per-architecture manifest digest) preserves arm64/amd64 portability.

To re-resolve the digest at fix time (do not blindly trust the value above; verify it is
still the current `:debian` index digest, or intentionally pin to the latest):
```bash
docker buildx imagetools inspect mcr.microsoft.com/devcontainers/base:debian | grep -i '^Digest'
```

Keep all other lines unchanged.

## Acceptance criteria
- [ ] `.devcontainer/ha-core/Dockerfile:1` references the base image as `image@sha256:<digest>` (digest present), not a bare tag.
- [ ] The image path is `mcr.microsoft.com/devcontainers/base` (legacy `vscode/` segment removed).
- [ ] A trailing comment records the human-readable tag and the date pinned (e.g. `# debian, pinned 2026-06-02`).
- [ ] The pinned digest matches what `docker buildx imagetools inspect mcr.microsoft.com/devcontainers/base:debian` returns for the index `Digest` (or a deliberately chosen, documented digest).
- [ ] The HA Core Functional Test container still builds: `docker compose -f .devcontainer/ha-core/docker-compose.yml build ha-dev` succeeds.
- [ ] No other lines in the Dockerfile are modified.

## Notes / risks
- **Scope boundary:** This ticket covers only the base `FROM` image. The same file also pins
  uv via `COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv` (line 28), which is
  itself unpinned and violates DOCKER-002/UV-001 — that belongs to a separate ticket; do not
  fold it in here. (The sibling Dockerfile already pins uv to `0.11.18@sha256:...` as the model.)
- **Related rule DOCKER-005** (`# syntax=docker/dockerfile:1` directive) is also missing from
  this file but absent the sibling's directive. Out of scope here; flag/track separately.
- **No auto-updater:** Because there is no Renovate/Dependabot in the repo, the digest will go
  stale and must be bumped manually. The trailing tag+date comment is the agreed mitigation.
  If a future ticket adds Renovate/Dependabot, configure it to bump the tag+digest as a literal
  pair on this line.
- **Index vs manifest digest:** Use the top-level index digest (multi-arch) shown above, not one
  of the per-platform child digests, so the dev container still works on both amd64 and arm64 hosts.
- **Verification needs registry access:** The build-check acceptance step requires network access
  to `mcr.microsoft.com`; run it where the registry is reachable.
