# BP-CON-08: Add the even-better-toml extension to both dev containers

> **Status: DONE** — `tamasfe.even-better-toml` added to both `devcontainer.json` extension arrays; both files still parse as valid JSONC.

**Severity:** low  |  **Rule:** DEVC-009 (containers-best-practices)  |  **Area:** `.devcontainer/devcontainer.json`, `.devcontainer/ha-core/devcontainer.json`

## Context
DEVC-009 says a dev container should declare (via `customizations.vscode.extensions`)
any tooling the project assumes, so the editor state matches the container state and
teammates aren't left without a language server/formatter/syntax support that the
project relies on.

This repo edits `pyproject.toml` (the project's full build/dependency/ruff/pytest
config lives there), yet neither dev container declares a TOML extension. VS Code has
no built-in TOML language support, so a teammate who opens either dev container gets
no syntax highlighting, validation, or formatting when touching `pyproject.toml`.

Scope note on the evidence: the finding also cites `hacs.json`, but that file is JSON,
not TOML — JSON support is built into VS Code, so it does not motivate this change. The
only TOML file in the repo is `pyproject.toml`:

```
$ find . -maxdepth 2 -name "*.toml"
./pyproject.toml
```

This is a **low**-severity, optional ergonomics improvement, not a correctness or
security issue. Per the rule's own recommendation, the fix is only worth doing if TOML
editing support is desired; if the team is content editing `pyproject.toml` without an
extension, closing this as won't-fix is acceptable.

## Current state
Both dev containers list extensions but omit any TOML extension.

`.devcontainer/devcontainer.json:33-37`:
```jsonc
"extensions": [
    "anthropic.claude-code",
    "ms-python.python",
    "charliermarsh.ruff"
],
```

`.devcontainer/ha-core/devcontainer.json:27-34`:
```jsonc
"extensions": [
    "anthropic.claude-code",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.debugpy",
    "charliermarsh.ruff",
    "redhat.vscode-yaml"
],
```

## Proposed fix
Add `tamasfe.even-better-toml` to the `extensions` array in both files. This is the
extension the DEVC-009 reference itself recommends. Keep alphabetical-ish/grouped order
consistent with each file.

`.devcontainer/devcontainer.json` — after:
```jsonc
"extensions": [
    "anthropic.claude-code",
    "ms-python.python",
    "charliermarsh.ruff",
    "tamasfe.even-better-toml"
],
```

`.devcontainer/ha-core/devcontainer.json` — after:
```jsonc
"extensions": [
    "anthropic.claude-code",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.debugpy",
    "charliermarsh.ruff",
    "redhat.vscode-yaml",
    "tamasfe.even-better-toml"
],
```

No other changes are required — no `settings` entries are needed for basic TOML support,
and nothing else in either file references TOML.

## Acceptance criteria
- [ ] `tamasfe.even-better-toml` appears in the `extensions` array of `.devcontainer/devcontainer.json`.
- [ ] `tamasfe.even-better-toml` appears in the `extensions` array of `.devcontainer/ha-core/devcontainer.json`.
- [ ] Both files remain valid JSONC (trailing-comma/brace structure intact); no other lines changed.
- [ ] No changes made for `hacs.json` (it is JSON, not TOML).

## Notes / risks
- Optional finding: if TOML editing support is explicitly not wanted, this ticket may be
  closed as won't-fix with a one-line rationale — no fix should be invented.
- Adding the extension only affects editors that build from these dev containers; it has
  zero runtime/container-image impact and cannot break CI or tests.
- Related: DEVC-009 also covers declaring `settings`; both files already declare the
  Python interpreter path and formatter settings, so no additional settings work is in
  scope here.
