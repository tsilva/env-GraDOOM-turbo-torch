---
name: build-release
description: Build, audit, publish, monitor, or verify a GraDOOM Python release. Use when the user asks to build release artifacts, cut/tag/publish a release, requests a specific GraDOOM version, invokes $build-release, diagnoses release packaging, or asks whether a version is live on PyPI.
---

# Build Release

Use the repository-owned release path and preserve the distinction between a
local candidate and an external publication. A local candidate is reversible;
pushing a release tag or publishing to PyPI is not.

GraDOOM currently has no checked-in release script or GitHub release workflow.
Build and audit local candidates with this skill, but stop publication requests
at that missing gate. Do not compensate with a hand-created tag, GitHub Release,
PyPI token, or manual upload. If repository-owned trusted-publishing automation
is added later, inspect it and update this skill before publishing.

Use normal PEP 440 project versions from `pyproject.toml`. Keep that version
identical to `src/gradoom/__init__.py` and the root `gradoom` entry in `uv.lock`.
Do not infer or write a version bump: require the user to choose it when the
checked-in version is not the intended release.

## Build a local candidate

1. Read `AGENTS.md` and use `$specs-author` as required there.

2. Confirm the worktree state and version without mutating either:

```bash
git status --short --branch
python3 .codex/skills/build-release/scripts/release_build.py check-version
```

Dirty files do not prevent an explicitly requested local candidate, but report
that it is not eligible for publication and preserve every existing change.

3. Run the locked source gates:

```bash
uv sync --frozen --group dev
.venv/bin/ruff check .
.venv/bin/pytest
```

Do not build a candidate when a source gate fails.

4. Confirm that the exact version is unused on PyPI:

```bash
python3 .codex/skills/build-release/scripts/release_build.py \
  check-pypi --version <version>
```

For packaging diagnosis of an already-published version, skip only this check
and say why. Never overwrite or republish an existing PyPI version.

5. Build into a fresh version-scoped directory:

```bash
.venv/bin/python .codex/skills/build-release/scripts/release_build.py build \
  --version <version> --out-dir dist/release-v<version>
```

The helper uses `uv build --no-sources`, requires exactly one universal wheel
and one source distribution, audits their metadata and contents, imports the
wheel in an isolated working directory using the locked environment, and prints
SHA-256 digests. It refuses to reuse an output directory so stale artifacts
cannot enter the candidate.

6. Report the two artifact paths, their SHA-256 digests, the exact version, and
every completed gate. Preserve failed artifacts and exact error output for
diagnosis.

## Publish or cut a release

Require all of the following before any tag or publication action:

- a clean worktree on the current branch;
- the branch synchronized with its configured upstream;
- an explicitly selected version matching all three metadata locations;
- an unused version on PyPI;
- a passing local candidate build from the exact commit; and
- a checked-in trusted-publishing workflow whose tag, artifact, audit, PyPI,
  and GitHub Release contract can be verified from repository source.

The last requirement is currently absent. Stop and report that
`.github/workflows/release.yml` (or an equivalent repository-owned workflow)
does not exist. Do not create or switch branches, synthesize release notes,
tag, push, or publish unless the user separately asks to add the missing release
infrastructure.

Never print, commit, or pass PyPI credentials on a command line. Trusted
publishing is the only acceptable normal PyPI publication path.

## Verify a published release

When publication infrastructure exists and a release is launched, monitor its
exact tag commit through the matching workflow. A workflow success is not the
final success signal: poll PyPI until files exist for the exact version, then
confirm the GitHub Release and artifact set.

Use:

```bash
release_sha="$(git rev-list -n 1 v<version>)"
gh run list --workflow release.yml --commit "$release_sha" --limit 5 \
  --json databaseId,status,conclusion,event,headBranch,headSha,displayTitle,url
gh run watch <run-id> --exit-status
```

If the workflow fails, inspect only failed logs with
`gh run view <run-id> --log-failed`. Do not manually replay the upload.

Confirm the exact PyPI version at:

```text
https://pypi.org/project/gradoom/<version>/
```

## Final response

For a local candidate, lead with the artifact directory and report both files,
digests, version, and gates. For a published release, lead with the exact PyPI
version URL and report the tag, workflow URL and conclusion, GitHub Release URL,
and every distribution filename. On failure, report the exact failed command or
gate and the next safe recovery action.
