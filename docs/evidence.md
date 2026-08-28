# Evidence command and v1 contract

`gradoom-evidence` is the installed, public entry point for GraDOOM parity and training evidence.
The first complete workflow is `parity_readiness`. It can exercise the evidence seam with fixture
inputs before the real WAD profile, reference provider, and pretrained policy corpus are available.
It reports those prerequisites as unavailable; it does not issue a parity certificate or make a
performance claim.

The command also provides the `development_training_benchmark` workflow. It is the inexpensive
integration benchmark: it drives the standalone GPU-resident trainer from a fresh policy and
optimizer, evaluates predeclared checkpoints on GraDOOM, and records its complete result using the
same evidence envelope. Development benchmark evidence is permanently non-authoritative and
claim-ineligible, even when every seed passes or no current parity certificate exists.

Run the repository fixture with:

```bash
gradoom-evidence \
  --manifest tests/fixtures/evidence/readiness-manifest.json \
  --output readiness-report.json
```

The command exits with status 2 and a concise error when the manifest is malformed, its schema
version is unsupported, a declared file does not match its SHA-256 digest, or a merge report is
incompatible. It writes the report only after all validation succeeds.

## Development training benchmark

A `development_training_benchmark` manifest uses the common `schema_version`, `workflow`,
`evidence_level`, `fixture`, `code_provenance`, and `declared_inputs` fields. Its `benchmark` object
contains:

- `training_seeds`: an optional non-empty array of unique uint32 cold-start seeds. Omitting it uses
  the predeclared protocol default `[123]`; development cohorts may contain fewer than five seeds.
- `failure_budget_steps`: the positive, predeclared maximum training step for each attempt.
- `checkpoint_steps`: unique increasing positive steps to evaluate, ending exactly at the failure
  budget. No checkpoint after the first passing checkpoint is trained or evaluated.
- `evaluation_episode_seeds`: exactly 100 unique predeclared uint32 GraDOOM game seeds, reused in
  the same order for every checkpoint and every training seed.
- `evaluation_action_seed`: the optional predeclared stochastic-policy RNG seed, defaulting to
  `123`.
- `trainer.command` and `trainer.arguments`: the standalone trainer invocation and fixed recipe,
  asset, and runtime arguments. The evidence command owns all seed, initialization, resume,
  checkpoint, timing, metrics, and evaluation arguments and rejects attempts to override them.
- `artifacts_directory`: the directory under which the run-identity and per-seed artifacts are
  durably retained.
- `bootstrap_artifacts`: optional one-time exclusions declared before the cohort. Every entry names
  a persistent read-only regular file outside the benchmark artifact directory, its SHA-256,
  creation elapsed seconds, creation protocol, exact reuse conditions, and true
  `persistent`/`run_independent`/`reused_unchanged` assertions. `contains_state` must explicitly set
  `learned`, `optimizer`, `rollout`, `seed_specific`, and `candidate_specific` to false. Missing,
  writable, linked, changed, incompletely disclosed, or state-bearing artifacts are rejected.
- `parity_certificate`: the current certificate availability and, when unavailable, its reason.
  An unavailable certificate is recorded as a claim-ineligibility reason but does not prevent a
  development run.
- `wad_profile`: required for non-fixture runs and validated through the immutable
  `freedoom2-deathmatch-v1` binding before training. The trainer and evaluator must report the same
  IWAD and PWAD hashes. CPU-only fixture manifests may omit real WAD assets.

For each seed, the command starts the reusable timer before attempt setup and invokes the existing
`standalone-gradoom-deathmatch-ppo-v2` trainer contract at every checkpoint cadence. The first
segment receives no resume or learned initialization; later segments resume only the preceding
checkpoint from that same attempt. This preserves one cold-start policy and optimizer lineage while
including recurring process initialization, uncached or per-process compilation, graph capture,
warm-up, training, checkpoint evaluation, and durable checkpoint writing in measured elapsed time.
Bootstrap bytes are verified before training and again after the cohort; their disclosed creation
cost is reported but excluded from reusable time. The output report cannot alias a bootstrap file.
The standalone trainer rounds an unaligned request down to the preceding complete-rollout boundary,
never up. A benchmark segment is accepted only when that execution boundary equals its exact
predeclared checkpoint; otherwise it fails closed without evaluation. The final report output must
not alias any generated checkpoint, seed file, or metrics file through an equal path, symlink, or
hardlink.

Evaluation uses the same trainer executable with `--evaluate-checkpoint`,
`--evaluation-episodes 100`, `--evaluation-seeds-file`, and stochastic actions. Every evaluator must
return all 100 episode records in the predeclared order. Passage is computed solely from their mean
`player_killcount` and requires at least `30.0`; separately reported `compatibility_killcount` is a
diagnostic and cannot change the verdict. The passing checkpoint, metrics, all preceding checkpoint
outcomes, all episode outcomes, and every process or evidence failure are retained and hashed in the
report evidence index.

The report has `authoritative: false` and `claim_eligible: false` unconditionally. Its per-seed
attempt states are `succeeded`, `exhausted`, `crashed`, `interrupted`, `evaluation_failed`, or
`evidence_failed`. The workflow status is `passed` only when every predeclared development attempt
succeeds; either status remains ineligible for public claims.

An interrupted trainer may leave a recovery checkpoint only when its metrics prove that policy,
optimizer, RNG, and progress state are all restorable and the checkpoint carries the exact run and
attempt identities. The report retains a hashed recovery journal binding that checkpoint, progress,
and accumulated reusable elapsed time. A later `--merge` resumes the same attempt, adds new elapsed
time to the journaled elapsed time, and preserves the original cold-start identity. Completed and
failed seeds are reused unchanged and are never rerun or replaced. Every attempt also has a durable
hashed state journal, so edited elapsed time, outcomes, failures, or recovery metadata cannot be
accepted as a completed unit.

`benchmark_protocol.continuation_identity` separately hashes the exact schema/trainer contract,
recipe, assets and bootstrap bytes, training/evaluation seeds, and timer phases/boundaries. Any
recipe, asset, seed, timer, schema, code provenance, WAD binding, evidence index, or artifact mismatch
fails before additional training work begins.

The standalone trainer's `--evaluation-seeds-file` accepts a UTF-8 JSON array of exactly 100 unique
uint32 game seeds. Its emitted config binds the complete ordered seed list, file hash, stochastic
action setting, `player_killcount` gate signal, and separate compatibility signal into the retained
evaluation evidence.

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
- `wad_profile` (required when `certified_freedoom2_wad_profile` is declared available): the
  `freedoom2-deathmatch-v1` profile ID and exactly one `gradoom` and one `env-vizdoom-turbo`
  provider binding. Each binding declares `iwad_path`, `pwad_path`, and the complete policy-facing
  `configuration`. Relative WAD paths are resolved from the manifest directory.

The repository fixture is the executable example of this schema:
[`tests/fixtures/evidence/readiness-manifest.json`](../tests/fixtures/evidence/readiness-manifest.json).

## First WAD profile

The immutable `freedoom2-deathmatch-v1` manifest binds:

- Freedoom2 IWAD SHA-256
  `a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b` and ViZDoom deathmatch
  PWAD SHA-256 `1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d`;
- `MAP01`, Doom skill 1, player mode, 320x240 source resolution, first-tic episode start,
  player-death termination, disabled HUD and screen flashes, and a 4200-tic truncation horizon;
- the custom discrete 17-action table, identified by SHA-256
  `0bd9dd28d67a88ef6bc54734f53d55bc4af597e672665a7f20d4b204098036af`, with frame skip 2; and
- a zero-filled bottom-32 mask, 84x84 area resize, the pinned RGB/area/GRAY8 conversion, CHW
  layout, and four-frame stacking.

The bundled JSON is a packaged representation, not its own authority. The evidence implementation
independently pins canonical profile SHA-256
`a3953ddfd4de7c8a99f51fed58dfbdc7002f6bf1c561ebbd25819aedf6e0cde7` and strictly checks every
key, type, and fixed value against the approved profile. A missing, undecodable, malformed,
incomplete, or semantically changed resource fails readiness before provider assets are inspected,
records `wad_profile_authority_failure`, and emits no profile or binding identity.

The command hashes all four provider asset paths before provider or training work. A missing or
incorrect asset, unequal provider bytes, or any configuration difference produces `status:
failed`, structured `wad_profile.failures`, and a `wad_profile_mismatch` claim reason in the
readiness report. Exact matches record the complete profile, both normalized provider bindings,
the bundled profile-manifest hash, and stable profile/binding identities. Asset paths are not
identity fields; their verified bytes and every policy-facing setting are.

These checks apply only to formal evidence. `GraDoomVecEnv` continues to accept compatible
operator-supplied Doom II and Freedoom IWADs for ordinary deathmatch use, where
`parity_certified` remains false and no WAD-profile evidence is inherited.

## Reference provider

Reference evaluation is bound to `env-ViZDoom-turbo` revision
`5b74973e4fbb1a96550a1884805b51fd6dcfe90f`. Install that immutable source revision into the
evaluation runtime:

```bash
uv pip install \
  'env-vizdoom-turbo @ git+https://github.com/tsilva/env-ViZDoom-turbo.git@5b74973e4fbb1a96550a1884805b51fd6dcfe90f#subdirectory=turbo'
```

The reference adapter verifies the installed Git commit from distribution provenance before it
imports the provider. Registry wheels, editable installs without verifiable Git provenance, and
all other revisions fail rather than silently running. Evaluation uses the current
`EnvViZDoomTurboVecEnv` export and explicitly requests both `player_killcount` and `killcount`.
Missing `player_killcount` is an error: it never falls back to compatibility `killcount`.

Episode records name `player_killcount` as the policy-quality outcome and retain
`compatibility_killcount` only as a separate diagnostic. GraDOOM and reference checkpoint reports
also record one provider-neutral execution identity containing the unchanged artifact hash,
checkpoint-frozen model/runtime contract, certified preprocessing hash, stochastic or diagnostic
argmax action mode, and an empty list of provider-specific modifications. Both providers load the
architecture, memory format, observation blur, frozen-encoder convolution mode, precision, policy
compilation, and float32 matrix-multiplication mode from the checkpoint rather than evaluator
defaults. The artifact is hashed before and after loading so a concurrently changed checkpoint is
rejected. The adapter does not offer observation correction, action remapping, fine-tuning, or
learned adaptation hooks.

## Report schema version 1

The JSON report records its schema version, workflow, evidence level, fixture state, readiness
status, claim eligibility and structured reasons, stable run identity, declared code provenance,
declared inputs, prerequisites, optional WAD-profile validation, and evidence index. A fixture
report is `unavailable` while real
prerequisites are missing, has `claim_eligible: false`, and names every missing prerequisite in
`claim_reasons`.

`run_identity` is the lowercase SHA-256 digest of canonical JSON containing the manifest schema,
workflow, evidence level, fixture state, code provenance, input names and declared hashes,
prerequisite identifiers, and—when supplied—the complete normalized WAD-profile binding. Input and
prerequisite ordering does not affect it. Canonical JSON uses
UTF-8, sorted object keys, no insignificant whitespace, and JSON separators `,` and `:`. File paths
are not identity fields; the declared content hashes are.

The `evidence_index.entries` array contains the raw manifest-file digest, every verified declared
input digest, and, when used, the bundled WAD-profile manifest plus every readable provider WAD
digest. `evidence_index.sha256` hashes the canonical JSON representation of that complete entries
array using the same rules. This makes index mutation detectable without attempting the impossible
operation of including the report's own digest inside itself.

## Safe continuation

Pass `--merge EXISTING_REPORT.json` when continuing evidence collection. The command validates the
existing report schema, recomputes its run identity from its recorded identity-bearing fields,
validates its evidence-index hash, then requires the recomputed identity to equal the run described
by the new manifest. Unlike code provenance, evidence levels, declared input hashes, or prerequisite
sets fail instead of being combined.

For `development_training_benchmark`, continuation additionally re-hashes every retained generated
artifact and validates every per-seed attempt journal. Only an `interrupted` attempt with a matching
recovery checkpoint and journal may execute more trainer work. Terminal successes and failures remain
attached to their original predeclared seed without replacement.
