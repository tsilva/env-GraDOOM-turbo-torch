---
name: build-release
description: Automatically version, build, audit, publish, monitor, and verify an env-gradoom-turbo-torch Python release through PyPI availability. Use when the user asks to build release artifacts, cut/tag/publish a release, requests a specific env-gradoom-turbo-torch version, invokes $build-release, diagnoses release packaging, or asks whether a version is live on PyPI.
---

# Build Release

Read and apply the shared `$release-workflow` skill at
`/Users/tsilva/.codex/skills/release-workflow/SKILL.md` before execution.
It owns common preflight, publication safeguards, `$push` integration,
workflow monitoring, verification, and reporting. The rules below are this
project's adapter; they retain its invocation default and required gates.
If the shared skill is unavailable, stop and report the missing dependency.

Use the repository-owned release path and preserve the distinction between a
local candidate and an external publication. A local candidate is reversible;
pushing a release tag or publishing to PyPI is not.

Treat an unqualified `$build-release` invocation as authorization to complete
the publication flow. Do not stop after building a local candidate: commit the
release metadata, tag and atomically push the release, monitor the exact
workflow, and verify the exact version on PyPI and GitHub. Use the Actions validation
flow only when the user explicitly asks for a candidate, dry run,
validation-only run, or no publication.

The repository publication path is `.github/workflows/release.yml`. A pushed
`v<version>` tag runs the locked source checks, builds and audits one universal
wheel and one source distribution, publishes them with PyPI trusted publishing,
and creates a GitHub Release. A `workflow_dispatch` run validates the same
source and artifacts but never publishes. Do not manually upload, substitute
artifacts, or replay only part of the workflow.

Use normal PEP 440 project versions from `pyproject.toml`. Keep that version
identical to `src/gradoom/__init__.py` and the root `env-gradoom-turbo-torch` entry
in `uv.lock`.
Automatic version selection always targets a final release. Promote an `aN`,
`bN`, `rcN`, or `.devN` checked-in version to its final base version; keep an
unused, untagged final version; otherwise increment the patch component until a
final version is unused on PyPI and untagged locally. Do not automatically
continue a prerelease series. A prerelease may be built or published only when
the user explicitly requests its exact version. `env-gradoom-turbo-torch` has no
upstream-derived `.postN` release scheme, so advance a `.postN` version to the
next final patch. Honor any exact user-selected final version as well.

## Validate in Actions without publication

Read `AGENTS.md` and apply `$specs-author`. Normal builds and release gates run
only in GitHub Actions. Fetch the configured upstream on main and resolve the
full pushed commit SHA, then dispatch:

```bash
gh workflow run release.yml --ref main -f ref=<full-pushed-main-sha>
```

The runner checks the locked environment, three matching versions, source lint
and tests, exact wheel/sdist contents, and isolated wheel import. Monitor the
exact SHA, download `release-v<version>`, and audit the existing artifacts:

```bash
python3 .codex/skills/build-release/scripts/release_build.py audit \
  --version <version> --dist-dir <download-directory>/primary
```

Compare downloaded hashes with the runner log. This dispatch never changes a
version, tags, publishes, or updates GradLab. Dirty local files are excluded
from the pushed source and must not be described as tested. The existing helper
`build` command remains available only for explicitly requested local diagnosis.

## Publish or cut a release

Require all of the following before any tag or publication action:

- a clean worktree on the current branch;
- the branch synchronized with its configured upstream;
- an automatically selected or explicitly requested version matching all three
  metadata locations;
- an unused version on PyPI;
- matching metadata and a passing lock consistency check; and
- a checked-in trusted-publishing workflow whose tag, artifact, audit, PyPI,
  and GitHub Release contract can be verified from repository source.

Start clean, fetch the configured upstream and release tags, require a
synchronized main branch, and run the metadata-only helper `prepare-version
--write`, `check-version`, and `check-pypi`. Run `uv lock --check --config-file
uv.toml` and `git diff --check`. Do not synchronize a local environment, run
source tests, or build local artifacts. If version preparation changed metadata, commit exactly
`pyproject.toml`, `src/gradoom/__init__.py`, and `uv.lock` as
`Release v<version>`. Verify the three committed versions agree. Actions validates and builds the
exact tagged commit before publication. Create an annotated tag only after every requirement
passes, then atomically push the current branch and tag:

```bash
git tag -a v<version> -m "Release v<version>"
git push --atomic origin HEAD v<version>
```

Release notes follow the shared workflow policy and are generated by the
GitHub Release job. Verify the checked-in workflow contract before tagging;
use shared stop conditions if it is absent or has changed.

## Verify a published release

Follow the shared monitoring and verification procedure for the `release.yml`
tag-push run at the full `v<version>` commit SHA. A `workflow_dispatch` run
validates artifacts but never publishes. Verify PyPI project `env-gradoom-turbo-torch` and
the GitHub Release for the same tag.

Require exactly one universal wheel and one source distribution on PyPI and
the matching GitHub Release.

## Update GradLab after successful publication

After the release succeeds and the exact PyPI version and required GitHub
Release artifacts pass external verification, update GradLab to consume the
latest successfully published `env-gradoom-turbo-torch` version. Complete
this step as part of the full publication flow; local builds, dry runs, and
inspection-only requests do not trigger it.

Read `/Users/tsilva/repos/tsilva/gradlab/AGENTS.md` and its required
specifications before editing. Synchronize GradLab's current branch with its
configured upstream and preserve existing work. Update every matching exact
pin in `pyproject.toml`, including platform-specific project dependencies and
the `train-runtime` dependency group. Use the just-verified release version;
if GradLab already consumes a newer verified publication, do not downgrade it.
Regenerate `uv.lock` with `uv lock --upgrade-package env-gradoom-turbo-torch`,
preserving unrelated pins, supply-chain constraints, and existing per-package
release-age exceptions. Review the dependency diff, validate lock consistency,
and run GradLab's relevant provider compatibility checks.

Report the GradLab version/pin and lockfile update separately from release
success. If synchronization, resolution, or validation fails, preserve the
published release and report the downstream update as incomplete with its
blocker; do not repeat publication.
