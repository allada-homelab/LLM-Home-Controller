# BP-UV-03: Add `[tool.uv] required-version` to pin a resolver floor

> **Status: DONE** — shipped in commit `0582f6e`. `[tool.uv] required-version = ">=0.8.0"` added to `pyproject.toml`; `uv lock --check` confirms no re-resolution.

**Severity:** medium  |  **Rule:** UVP-008 (uv-best-practices)  |  **Area:** `pyproject.toml`, dev container / CI uv toolchain

## Context
`pyproject.toml` has no `[tool.uv]` table at all, so there is no declared minimum uv
version. This project is not a solo prototype — it has multiple uv binaries in play that
must agree on resolver behavior:

- `.devcontainer/Dockerfile:17` pins `ghcr.io/astral-sh/uv:0.11.18@sha256:78bc42…` and
  `.devcontainer/devcontainer.json:30` runs `uv sync` as its `postCreateCommand`.
- `.devcontainer/ha-core/Dockerfile:28` installs `ghcr.io/astral-sh/uv:latest` — an
  unpinned, floating binary that can drift to any newer version.
- Any contributor running `uv sync` / `uv lock` on their own machine uses whatever uv
  they happen to have installed.

Without a `required-version` floor, version skew between these silently changes resolver
output (`uv.lock` churn that looks like an unintended dependency bump but is really two
resolvers disagreeing) and silently ignores newer config fields on old binaries.

This repo makes the risk concrete: the committed `uv.lock` is `version = 1, revision = 3`.
The `revision = 3` lockfile format is emitted by recent uv (0.8.x line); a contributor on
an old uv (e.g. the skill's bare baseline 0.5.14) cannot correctly read or round-trip this
lockfile. A floor declared in `pyproject.toml` makes the requirement explicit and
machine-checkable: newer uv refuses to operate below the floor; older uv ignores the field
(the worst case stays the status quo, never worse).

## Current state
`pyproject.toml` (full file, 47 lines) contains `[project]`, `[dependency-groups]`,
`[tool.pytest.ini_options]`, `[tool.ruff]`, and `[tool.ruff.lint*]` — but no `[tool.uv]`.

`pyproject.toml:1-22`:
```toml
[project]
name = "llm-home-controller"
version = "0.1.0"
description = "HomeAssistant Conversation Agent for OpenAI API-compatible LLM endpoints"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "homeassistant",
    "aiohttp",
    "voluptuous",
]

[dependency-groups]
dev = [
    "hassil>=3.0",
    ...
]
```

Supporting evidence:
- `uv.lock:1-2` → `version = 1` / `revision = 3` (lockfile format requires recent uv).
- `.devcontainer/Dockerfile:17` → uv pinned at `0.11.18`.
- `.devcontainer/ha-core/Dockerfile:28` → uv `:latest` (floating).
- `.devcontainer/devcontainer.json:30` → `"postCreateCommand": "uv sync"`.

## Proposed fix
Add a `[tool.uv]` table to `pyproject.toml`. The skill's bare baseline for the field is
`>=0.5.14`, but that floor is inconsistent with this repo's own `revision = 3` lockfile and
its pinned `0.11.18` toolchain. Pin the floor to a version that can actually read the
committed lockfile and aligns with the dev container. `>=0.8.0` is the conservative choice
(first uv that emits `revision`-bearing lockfiles in this line); the dev container's
`0.11.18` already satisfies it.

Insert after the `[dependency-groups]` block (around line 23), before
`[tool.pytest.ini_options]`:

```toml
[tool.uv]
required-version = ">=0.8.0"  # floor for the committed revision=3 uv.lock; dev container ships 0.11.18
```

Before (`pyproject.toml:22-24`):
```toml
    "ruff>=0.9",
]

[tool.pytest.ini_options]
```

After:
```toml
    "ruff>=0.9",
]

[tool.uv]
required-version = ">=0.8.0"  # floor for the committed revision=3 uv.lock; dev container ships 0.11.18

[tool.pytest.ini_options]
```

No other change is in scope. Do not touch the Dockerfiles in this ticket (see Notes).

## Acceptance criteria
- [ ] `pyproject.toml` contains a `[tool.uv]` table with a `required-version` key.
- [ ] The pinned floor is satisfiable by the dev container's pinned uv (`0.11.18`) and is
      high enough to read the committed `uv.lock` (`revision = 3`) — i.e. `>= 0.8.0`.
- [ ] `uv sync --locked` (or `uv lock --check`) succeeds and produces no `uv.lock` diff,
      confirming the floor does not force a re-resolution.
- [ ] `uv --version` in the dev container reports a version `>=` the declared floor (no
      "required-version" error on any uv command).

## Notes / risks
- The skill text recommends `>=0.5.14` as the *field-introduction* baseline. Using it here
  would be self-contradictory because the repo's `uv.lock` is `revision = 3`, unreadable by
  uv 0.5.x. Hence the deliberate bump to `>=0.8.0`. If a reviewer disagrees, the only safe
  downward bound is "lowest uv that reads `revision = 3`"; do not drop below that.
- Related follow-up (out of scope, separate ticket): `.devcontainer/ha-core/Dockerfile:28`
  uses `uv:latest`. A floating tag can drift *above or below* expectations and undercuts the
  whole point of a declared floor; it should be pinned to a digest like the main
  `.devcontainer/Dockerfile` already does (UV-001 / DEVC pinning rules). `required-version`
  enforces a *minimum*, not the exact toolchain, so pinning the image is the complementary
  control.
- Ordering: independent of other tickets. Safe to land alone. Editing the TOML table
  placement is cosmetic — uv reads `[tool.uv]` regardless of position; placing it adjacent
  to the other `[tool.*]` tables just keeps the file tidy.
- Zero runtime/test impact: this field only gates uv-binary behavior, not the package's
  Python dependencies or HomeAssistant integration.
