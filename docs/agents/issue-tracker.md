# Issue tracker: GitHub

Issues and specs for this repository live as GitHub issues in `tsilva/env-Doom-turbo-torch`. Use the `gh` CLI for all operations.

## Conventions

- Create an issue with `gh issue create --title "..." --body "..."`.
- Read an issue with `gh issue view <number> --comments`.
- List issues with `gh issue list`, using appropriate state and label filters.
- Comment with `gh issue comment <number> --body "..."`.
- Apply or remove labels with `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Close an issue with `gh issue close <number> --comment "..."`.
- Infer the repository from the configured Git remote.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill routing

- When a skill says “publish to the issue tracker,” create a GitHub issue.
- When a skill says “fetch the relevant ticket,” run `gh issue view <number> --comments`.
- Resolve a bare issue or pull-request number by trying `gh pr view <number>` and then `gh issue view <number>`.
