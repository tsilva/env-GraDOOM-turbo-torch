---
name: build-release
description: Automatically version, build, audit, publish, monitor, or verify a GraDOOM Python release. Use when the user asks to build release artifacts, cut/tag/publish a release, requests a specific GraDOOM version, invokes $build-release, diagnoses release packaging, or asks whether a version is live on PyPI.
---

# Build Release

Use the repository-owned release path and preserve the distinction between a
local candidate and an external publication. A local candidate is reversible;
pushing a release tag or publishing to PyPI is not.

The repository publication path is `.github/workflows/release.yml`. A pushed
`v<version>` tag runs the locked source checks, builds and audits one universal
wheel and one source distribution, publishes them with PyPI trusted publishing,
and creates a GitHub Release. A `workflow_dispatch` run validates the same
source and artifacts but never publishes. Do not manually upload, substitute
artifacts, or replay only part of the workflow.

Use normal PEP 440 project versions from `pyproject.toml`. Keep that version
identical to `src/gradoom/__init__.py` and the root `gradoom` entry in `uv.lock`.
Treat an untagged, unused checked-in version as pending. Otherwise select the
next unused, untagged version automatically: increment the numeric suffix for
`aN`, `bN`, `rcN`, or `.devN` versions, and increment the patch component for a
final or `.postN` version. GraDOOM has no upstream-derived `.postN` release
scheme. Honor an exact user-selected version instead of auto-selection.

## Build a local candidate

1. Read `AGENTS.md` and use `$specs-author` as required there.

2. Confirm the worktree state and synchronized metadata without mutating either:

```bash
git status --short --branch
python3 .codex/skills/build-release/scripts/release_build.py check-version
```

Dirty files do not prevent an explicitly requested local candidate, but report
that it is not eligible for publication and preserve every existing change.

3. Select the release version. On a clean worktree, write an automatic bump when
the checked-in version is already tagged or published:

```bash
python3 .codex/skills/build-release/scripts/release_build.py \
  prepare-version --write
```

For an exact version explicitly requested by the user, add `--to <version>`.
The helper checks local tags and PyPI, skips occupied versions, and transactionally
updates `pyproject.toml`, `src/gradoom/__init__.py`, and the root `gradoom`
entry in `uv.lock`. If the worktree was dirty, run without `--write`; proceed
only when the reported pending version requires no bump. Never layer an
automatic version edit onto existing user changes.

4. Run the locked source gates after version preparation:

```bash
uv sync --frozen --group dev
.venv/bin/ruff check .
.venv/bin/pytest
```

Do not build a candidate when a source gate fails.

5. Confirm that the selected exact version is still unused on PyPI:

```bash
python3 .codex/skills/build-release/scripts/release_build.py \
  check-pypi --version <version>
```

For packaging diagnosis of an already-published version, skip only this check
and say why. Never overwrite or republish an existing PyPI version.

6. Build into a fresh version-scoped directory:

```bash
.venv/bin/python .codex/skills/build-release/scripts/release_build.py build \
  --version <version> --out-dir dist/release-v<version>
```

The helper uses `uv build --no-sources`, requires exactly one universal wheel
and one source distribution, audits their metadata and contents, imports the
wheel in an isolated working directory using the locked environment, and prints
SHA-256 digests. It refuses to reuse an output directory so stale artifacts
cannot enter the candidate.

7. Report the two artifact paths, their SHA-256 digests, the selected version,
whether metadata was bumped, and every completed gate. Preserve failed
artifacts and exact error output for diagnosis. A candidate with an uncommitted
automatic bump is not eligible for publication.

## Publish or cut a release

Require all of the following before any tag or publication action:

- a clean worktree on the current branch;
- the branch synchronized with its configured upstream;
- an automatically selected or explicitly requested version matching all three
  metadata locations;
- an unused version on PyPI;
- a passing local candidate build from the exact commit; and
- a checked-in trusted-publishing workflow whose tag, artifact, audit, PyPI,
  and GitHub Release contract can be verified from repository source.

Start clean, run `prepare-version --write`, and complete the source and candidate
gates. If version preparation changed metadata, commit exactly
`pyproject.toml`, `src/gradoom/__init__.py`, and `uv.lock` as
`Release v<version>`. Verify the committed tree is identical to the source used
for the passing candidate. Create an annotated tag only after every requirement
passes, then atomically push the current branch and tag:

```bash
git tag -a v<version> -m "Release v<version>"
git push --atomic origin HEAD v<version>
```

Do not create or switch branches, synthesize release notes, or move an existing
release tag. If the workflow is absent or no longer matches this contract, stop
before tagging and repair the repository-owned release path first.

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
