#!/usr/bin/env bash
# Point this clone's git at the repo-tracked hooks in .githooks/.
# Every collaborator must run this once after cloning.
#
# What it wires up:
#   commit-msg — refuses commits whose message contains AI-attribution
#                tags (Co-Authored-By Claude, Claude-Session, "Generated
#                with Claude Code", claude.ai links, etc.)
#   pre-push   — same check, applied to every commit being pushed, as a
#                safety net for commits made with --no-verify or from
#                another machine.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ ! -d .githooks ]; then
  echo "setup-git-hooks: .githooks/ is missing — run this from the repo root" >&2
  exit 1
fi

chmod +x .githooks/commit-msg .githooks/pre-push
git config core.hooksPath .githooks

echo "Git hooks installed."
echo "  core.hooksPath = $(git config core.hooksPath)"
echo "  hooks:         $(ls .githooks | tr '\n' ' ')"
