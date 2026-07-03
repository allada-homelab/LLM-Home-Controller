#!/usr/bin/env bash
# SessionStart hook — generic project grounding.
#
# stdout is injected as session context. Bootstraps the environment if needed
# and prints a one-line git status. Fails OPEN everywhere — every probe degrades
# to a no-op or warning, never aborts the session.

set -euo pipefail

root=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
[ -z "$root" ] && exit 0

# Bootstrap deps on first run (fail-open: a sync failure must not block the
# session). --locked so a stale environment never churns the committed uv.lock.
if [ ! -d "$root/.venv" ] && command -v uv >/dev/null 2>&1; then
    echo "- .venv missing — running 'uv sync --locked --all-groups'..."
    uv sync --locked --all-groups --project "$root" >/dev/null 2>&1 \
        || echo "  WARNING: uv sync failed; run 'uv sync --locked --all-groups' manually."
fi

# One-line git status: branch + dirty-file count.
branch=$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
dirty=$( { git -C "$root" status --porcelain 2>/dev/null || true; } | wc -l | tr -d ' ')
echo "- git: branch '$branch', $dirty uncommitted file(s)."

exit 0
