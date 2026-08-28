# Evidence command and v1 contract

`gradoom-evidence` is the installed, public entry point for GraDOOM parity and training evidence.
The first complete workflow is `parity_readiness`. It can exercise the evidence seam with fixture
inputs before the real WAD profile, reference provider, and pretrained policy corpus are available.
It reports those prerequisites as unavailable; it does not issue a parity certificate or make a
performance claim.

Run the repository fixture with:

```bash
gradoom-evidence \
  --manifest tests/fixtures/evidence/readiness-manifest.json \
  --output readiness-report.json
```

The command exits with status 2 and a concise error when the manifest is malformed, its schema
version is unsupported, a declared file does not match its SHA-256 digest, or a merge report is
incompatible. It writes the report only after all validation succeeds.

## Manifest schema version 1

The v1 readiness manifest is a JSON object with these fields:

- `schema_version`: the integer `1`.
- `workflow`: `parity_readiness`.
- `evidence_level`: `development`. Development evidence is always non-authoritative.
- `fixture`: whether fixture inputs are in use. Fixture evidence is never claim-eligible.
- `code_provenance`: `repository`, `revision`, and boolean `dirty` state declared by the operator.
- `declared_inputs`: uniquely named files, each with a path and lowercase SHA-256 digest. Relative
  paths are resolved from the manifest directory and are verified before report creation.
- `prerequisites`: uniquely identified readiness prerequisites with an `available` boolean. Every
  unavailable prerequisite must include a human-readable `reason`.

The repository fixture is the executable example of this schema:
[`tests/fixtures/evidence/readiness-manifest.json`](../tests/fixtures/evidence/readiness-manifest.json).

## Report schema version 1

The JSON report records its schema version, workflow, evidence level, fixture state, readiness
status, claim eligibility and structured reasons, stable run identity, declared code provenance,
declared inputs, and evidence index. A fixture report is `unavailable` while real prerequisites are
missing, has `claim_eligible: false`, and names every missing prerequisite in `claim_reasons`.

`run_identity` is the lowercase SHA-256 digest of canonical JSON containing the manifest schema,
workflow, evidence level, fixture state, code provenance, input names and declared hashes, and
prerequisite identifiers. Input and prerequisite ordering does not affect it. Canonical JSON uses
UTF-8, sorted object keys, no insignificant whitespace, and JSON separators `,` and `:`. File paths
are not identity fields; the declared content hashes are.

The `evidence_index.entries` array contains the raw manifest-file digest and every verified declared
input digest. `evidence_index.sha256` hashes the canonical JSON representation of that complete
entries array using the same rules. This makes index mutation detectable without attempting the
impossible operation of including the report's own digest inside itself.

## Safe continuation

Pass `--merge EXISTING_REPORT.json` when continuing evidence collection. The command validates the
existing report schema and evidence-index hash, then requires its `run_identity` to equal the run
described by the new manifest. Unlike code provenance, evidence levels, declared input hashes, or
prerequisite sets fail instead of being combined.
