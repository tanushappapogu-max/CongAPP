# Commit policy

**No AI-assistant attribution on any commit, ever.**

Every commit must be authored by a human team member. Do not add any
of the following to a commit message under any circumstances:

- `Co-Authored-By: Claude ...`
- `Co-Authored-By: <anything>@anthropic.com`
- `Claude-Session: ...`
- `🤖 Generated with Claude Code`
- `Generated with Claude Code` (or any variation)
- Links to `claude.ai/code` or `claude.ai/chat`
- `Co-Authored-By: Copilot ...` or any similar AI-tool attribution

## How this is enforced

Two git hooks live in `.githooks/` and are tracked in the repo:

- `commit-msg` — refuses the commit locally if the message contains a
  forbidden tag.
- `pre-push` — refuses the push if any commit in the range being pushed
  carries a forbidden tag (catches `--no-verify` commits and commits
  authored on another machine).

Point your local git at them after cloning:

```bash
./scripts/setup-git-hooks.sh
```

Or manually:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/*
```

## If the hook fires

Rewrite the offending commit message before continuing:

```bash
# most recent commit
git commit --amend

# older commit in the current branch
git rebase -i <sha-before-the-offender>
# mark the offender as "reword", save, edit the message, save.
```

Then re-run the push.
