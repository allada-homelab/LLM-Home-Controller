# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04@sha256:81380e4c9c14e8a629ff39029639e4b7893e67400246fa7782a0fe7dc193a02a

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# vim + make baked into the image so they're cached, not reinstalled on every
# container rebuild (DEVC-002). git/python/etc. come from devcontainer features.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        vim \
        make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv, pinned by version+digest (UV-001). The uv image is distroless, so this
# copies only the binaries. Replaces the unverified `curl | sh` install (DOCKER-026).
COPY --from=ghcr.io/astral-sh/uv:0.11.18@sha256:78bc42400d77b0678ba95765305c826652ed5431f399257271dda681d0318f03 /uv /uvx /bin/
