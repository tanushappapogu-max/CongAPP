# Agent instructions for this repo

If you are an AI coding assistant (Claude Code, Cursor, Codex, Aider,
GitHub Copilot, or anything similar) opening this repository, read the
whole file before you write any commit.

## Commit rule (hard requirement)

**Do not add AI-assistant attribution to any commit message. Ever.**

Forbidden — the commit-msg and pre-push hooks will refuse these:

- `Co-Authored-By: Claude ...`
- `Co-Authored-By: <anyone>@anthropic.com`
- `Co-Authored-By: Copilot ...`
- `Claude-Session: ...`
- Any line starting with the robot emoji plus "Generated with ..."
- Any URL under `claude.ai/code` or `claude.ai/chat`
- Anything else that identifies an AI tool as an author of the change

If you would normally add such a line by default, suppress it here.

Every commit must be attributable to the human who authored it. Use
that human's real name and email in `git config user.name` and
`git config user.email`.

## Why

This is a team repository under review. Attribution lines that flag
AI authorship make it look like commits were not read by a human
before landing. The team's policy is: humans own every commit.

## Setup after cloning

```bash
./scripts/setup-git-hooks.sh
```

That points `core.hooksPath` at `.githooks/`, which runs both:

- `.githooks/commit-msg` — refuses commits locally.
- `.githooks/pre-push` — refuses pushes with any offending commit in
  the range.

## If a hook fires

Rewrite the offender:

```bash
git commit --amend              # if it is the tip
git rebase -i <sha-before>      # if it is deeper; mark as "reword"
```

Then re-run the push.

## Everything else

You are free to help with code as usual — the only restriction is
what ends up in the commit message. Full policy in
[COMMIT_POLICY.md](COMMIT_POLICY.md).
