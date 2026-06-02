# BP-CON-01: Pin uv image to version+digest in the ha-core dev container Dockerfile

> **Status: WON'T DO (permanent).** Closed without action by maintainer decision.

**Severity:** high  |  **Rule:** UV-001 (containers-best-practices)  |  **Area:** `.devcontainer/ha-core/Dockerfile`

## Context
The "HA Core Functional Test" dev container builds `uv` into its image by copying the binary out of Astral's distroless uv image. It references the image as `:latest`, an unpinned mutable tag. Every rebuild may silently pull a different `uv` build, so the dev environment is not reproducible: two engineers (or the same engineer on different days) can end up with different `uv` versions, and a future `uv` release could change behavior without any visible diff in the repo. Because `COPY --from=...:latest` resolves the tag at build time with no digest verification, there is also no guarantee the copied binary matches what was published when the line was written.

This repo already has the correct pattern in the sibling primary dev container Dockerfile (`.devcontainer/Dockerfile:17`), which pins the literal version+digest pair. The ha-core Dockerfile is simply out of sync with it. UV-001 (with SEC-010 on pinning) calls for pinning the literal version and digest as a single pair so builds are reproducible and a tool like Renovate can bump the pair together.

This is a real issue, not a false positive: the line is live, used by the documented functional-test workflow in `CLAUDE.md`, and the unpinned tag is the only `uv` reference in this file.

## Current state
`.devcontainer/ha-core/Dockerfile:27-28`:

```dockerfile
# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
```

For reference, the already-correct sibling at `.devcontainer/Dockerfile:17`:

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.11.18@sha256:78bc42400d77b0678ba95765305c826652ed5431f399257271dda681d0318f03 /uv /uvx /bin/
```

## Proposed fix
Pin the same version+digest pair already used by the main Dockerfile. Minimal one-line edit at `.devcontainer/ha-core/Dockerfile:28`.

Before:
```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
```

After:
```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.11.18@sha256:78bc42400d77b0678ba95765305c826652ed5431f399257271dda681d0318f03 /uv /usr/local/bin/uv
```

Keep the existing destination (`/usr/local/bin/uv`) and the single-binary copy as-is — this finding is only about pinning the source image, not about changing the install layout. Do not add `/uvx` or alter the destination unless a separate ticket calls for it.

## Acceptance criteria
- [ ] `.devcontainer/ha-core/Dockerfile` no longer references `ghcr.io/astral-sh/uv:latest`.
- [ ] The `COPY --from` source is pinned to a literal `VERSION@sha256:DIGEST` pair.
- [ ] The pinned version+digest matches `.devcontainer/Dockerfile:17` (`0.11.18@sha256:78bc42400d77b0678ba95765305c826652ed5431f399257271dda681d0318f03`) so both dev containers use the same uv.
- [ ] `grep -rn "astral-sh/uv:latest" .devcontainer/` returns no matches.
- [ ] The ha-core dev container still builds and `uv --version` reports `0.11.18` inside it.

## Notes / risks
- Related rules: SEC-010 (pin images by digest). The main Dockerfile already follows both UV-001 and SEC-010; this ticket only brings ha-core into line.
- Keep the version+digest as a single matched pair — bumping the version without updating the digest (or vice versa) will fail the pull. If/when Renovate is configured, point it at both Dockerfiles so the pair stays in sync across them.
- Low blast radius: the change affects only the functional-test dev container image build, not runtime HA behavior or the published integration. The image must be rebuilt (not just `--skip-pip` restart) for the new pin to take effect.
- Verify the digest is still valid before merging if significant time has passed; if Astral has rotated it, re-pin both Dockerfiles together rather than leaving them divergent.
