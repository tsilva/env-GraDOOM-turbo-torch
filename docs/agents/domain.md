# Domain Docs

Engineering skills must consume this repository’s domain documentation before exploring or proposing changes.

## Before exploring

Read:

- `CONTEXT.md` at the repository root, when present.
- Relevant system-wide ADRs under `docs/adr/`, when present.

If these files do not exist, proceed silently. Domain-modeling workflows create them when durable terminology or decisions emerge.

## Layout

This is a single-context repository:

- `CONTEXT.md` contains the project glossary and domain model.
- `docs/adr/` contains system-wide architectural decisions.
- `src/` contains the implementation.

## Vocabulary

Use terms exactly as defined in `CONTEXT.md` in issues, specifications, hypotheses, tests, and implementation discussions. Avoid synonyms that the glossary explicitly rejects.

If a required concept is absent, reconsider whether new terminology is necessary or flag the gap for domain modeling.

## ADR conflicts

Surface any conflict with an existing ADR explicitly rather than silently overriding it.
