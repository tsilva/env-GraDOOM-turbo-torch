## Product Specifications

At the start of substantive repository work, use `$specs-author` to read the entire root `SPECS.md` once per ongoing task. Reuse unchanged contents across follow-ups and composed skills. Reread when the file or repository association changes, or new stakeholder intent requires a fresh comparison. Before finishing, always check for new or changed stakeholder intent; reread if the file or intent changed. Purely conversational follow-ups do not require another full read.

- Treat `SPECS.md` as the persistent source of stakeholder requirements that cannot be inferred reliably from code or remembered conversations.
- Apply the scope test to proposed and existing requirements: root `SPECS.md` contains only project-wide intent; scoped intent belongs in its nearest authoritative specification and must not be broadened to fit the root.
- If the task, repository, or user request contradicts, omits, or ambiguously interprets the specification, tell the user. Continue safe exploration and work that does not depend on resolving the issue, but never silently choose an interpretation.
- Never edit `SPECS.md` from inference. Propose the exact change, explain why it reflects stakeholder intent, and edit the file only after the user explicitly approves that exact change.
- Keep `SPECS.md` complete, concise, and compacted. It must contain stakeholder intent rather than implementation, architecture, operations, or transient project detail.

## Agent skills

### Issue tracker

Issues and specs are tracked in GitHub Issues for `tsilva/env-GraDOOM-turbo-torch`. See `docs/agents/issue-tracker.md`.

For explicit AFK implementation of an approved root issue and its child issue graph, use `$codex-issue-orchestrate`. Start with `--dry-run`; the skill uses isolated worktrees, reviewed child PRs into a specification integration branch, and a final PR for human review.

### Triage labels

Use the canonical `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix` labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout with `CONTEXT.md` at the repository root and system-wide ADRs under `docs/adr/`. See `docs/agents/domain.md`.

## Release builds

Use the repository `$build-release` skill. Normal validation and publication
builds run only in GitHub Actions; local preparation handles metadata and Git.
