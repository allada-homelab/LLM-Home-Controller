# BP-PY-05: Remove stale egg-info artifact; do not declare a build backend for this HACS component

**Severity:** low  |  **Rule:** PY-004 (python-best-practices)  |  **Area:** `pyproject.toml`, `llm_home_controller.egg-info/` (repo root)

## Context
`pyproject.toml` has no `[build-system]` table (`grep -c build-system pyproject.toml` = 0). Under PY-004, a missing `[build-system]` means pip falls back to legacy `setuptools` with auto-discovery — the failure mode that shipped a stale `llm_home_controller.egg-info/` into this repo from a prior editable install.

However, the right fix here is NOT to add a build backend. This project is a **Home Assistant custom integration distributed via HACS**, not a pip package:

- `hacs.json` exists at the repo root (`render_readme`, `homeassistant: 2025.1.0`) — HACS is the distribution channel.
- `custom_components/llm_home_controller/manifest.json` defines the HA integration (`domain`, `config_flow`, `requirements`). HA loads the integration by copying/symlinking `custom_components/llm_home_controller/` into a config dir — there is no wheel install in the supported workflow.
- The project's own CLAUDE.md describes setup as a symlink of the integration into `/workspace/ha-config/custom_components/`, never `pip install .`.
- `README.md` has zero `pip install` / editable / setup.py references.

PY-004's "When NOT to apply" covers exactly this: the package is never built or pip-installed for distribution, so declaring a backend would be speculative configuration for a workflow that doesn't exist. The real defect is the leftover artifact, which is what produced the over-broad `top_level.txt` cited in the finding.

The `egg-info` directory is already gitignored (`.gitignore:12` = `*.egg-info/`) and untracked (`git ls-files | grep egg-info` returns nothing), so it never reached anyone else — but it is still present in this working tree and is stale/misleading.

## Current state
`pyproject.toml` (no `[build-system]`, starts straight at `[project]`):
```toml
# /workspace/LLM-Home-Controller/pyproject.toml:1
[project]
name = "llm-home-controller"
version = "0.1.0"
...
[dependency-groups]
dev = [ ... ]
```

Stale artifact at repo root, gitignored and untracked:
```
/workspace/LLM-Home-Controller/llm_home_controller.egg-info/
  PKG-INFO  SOURCES.txt  requires.txt  dependency_links.txt  top_level.txt
```

`top_level.txt` is wildly over-broad — it lists not just `custom_components`, `homeassistant`, `aiohttp`, `voluptuous` but essentially every installed site-package (`certifi`, `boto3`, `PIL`, `cffi`, `sqlalchemy`, ... ~150 entries). This is the PY-004 setuptools auto-discovery failure mode captured on disk.

## Proposed fix
Delete the stale artifact. Do not add a `[build-system]` table.

```bash
rm -rf /workspace/LLM-Home-Controller/llm_home_controller.egg-info
```

Leave `pyproject.toml` unchanged — it is consumed only as a project-config file (ruff, pytest, dependency-groups for the dev/test venv), not as a build manifest. `*.egg-info/` is already in `.gitignore:12`, so no gitignore change is needed.

Optional (only if a maintainer decides the repo SHOULD be pip-installable, e.g. for a future PyPI release): add the hatchling backend per PY-004. This is explicitly out of scope for this ticket and should not be done absent a concrete distribution requirement:
```toml
# only if pip-installable distribution is actually adopted
[build-system]
requires = ["hatchling>=1.26"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["custom_components/llm_home_controller"]
```

## Acceptance criteria
- [ ] `llm_home_controller.egg-info/` no longer exists in the working tree (`test ! -e llm_home_controller.egg-info && echo gone`).
- [ ] `pyproject.toml` still has no `[build-system]` table (no speculative backend added).
- [ ] `.gitignore` still contains `*.egg-info/` (line 12) so the artifact cannot reappear in a future commit.
- [ ] `uv run pytest tests/ -v` and `uv run ruff check .` still pass (confirms removal didn't disturb the dev/test setup).
- [ ] No new `pip install .` / editable-install step is introduced in docs or scripts.

## Notes / risks
- Risk is near-zero: the directory is untracked and gitignored, so deleting it changes nothing in git and nothing in the HA runtime (HA never reads egg-info).
- An editable install (`pip install -e .`) regenerates this directory and re-triggers the auto-discovery problem. Avoid editable installs for this repo; the supported dev loop is the symlink-into-`ha-config` workflow from CLAUDE.md.
- Related: this is the same "config-vs-build" distinction as the uv-best-practices rules — `pyproject.toml` here is legitimately a tool-config-only file with `[dependency-groups]` for the test venv, which does not require a build backend.
- If the optional pip-installable path is ever taken, that is a separate decision/ticket; pair it with a real release/CI workflow rather than adding `[build-system]` in isolation.
