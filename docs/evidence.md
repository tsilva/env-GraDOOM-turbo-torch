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
- `trainer.command`, `trainer.code_root`, and `trainer.arguments`: the standalone trainer invocation
  and fixed recipe,
  asset, and runtime arguments. The evidence command owns all seed, initialization, resume,
  checkpoint, timing, metrics, and evaluation arguments and rejects attempts to override them. The
  command must be one resolved Python interpreter plus one source file; native wrappers, shells,
  `-c`, `eval`/`exec`, dynamic runners, and subprocess indirection are rejected because their full
  executed-code closure cannot be proven. Capability-bearing modules use a positive access policy:
  only the exact configuration and fixture operations needed by the documented trainer are allowed;
  namespace dictionaries, reflection, computed attributes, container aliases, and equivalent
  process-replacement paths fail closed. Relative executables and scripts are resolved from the
  manifest directory. The entry point and its
  transitive local Python import closure below the code root are hashed into the recipe identity and
  reverified after the cohort, so same-path helper mutation or mid-cohort replacement fails closed.
- `artifacts_directory`: the directory under which the run-identity and per-seed artifacts are
  durably retained.
- `bootstrap_artifacts`: optional one-time exclusions declared before the cohort. Every entry names
  a persistent read-only regular file outside the benchmark artifact directory, its SHA-256,
  creation elapsed seconds, and exact canonical reuse conditions. Eligibility is not established by
  those operator-authored fields. The packaged `gradoom-time-authority create-bootstrap` operation
  exclusively creates and fsyncs the canonical artifact, measures its creation cost, and records its
  persistent object identity in a signed ledger. A later, separate
  `record-bootstrap-reuse` observation must see the same read-only object and bytes before
  `attest-bootstrap-reuse` can issue eligibility evidence. Formal validation reads that repository-
  owned authority's private ledger; first creation, deletion/recreation, copied paths, ledger
  rollback, state reset, and replayed signatures are ineligible. Eligible
  bytes use only the `gradoom-declarative-bootstrap-v1` canonical JSON contract. Compiler-target
  inputs reject seed, candidate, policy, optimizer, rollout, and learned state, and opaque outputs
  remain ineligible. Files are verified before and after the cohort.
- `elapsed_time_anchors`: required for every training seed before a run. Each training seed
  has a pre-attempt Ed25519-signed start record from an independently controlled authority. Fixture
  anchors are pinned to the public fixture key. Formal anchors must match the identity in the
  persistent state directory named by `GRADOOM_REUSABLE_TIME_AUTHORITY_STATE` and the separately
  retained monotonic witness named by `GRADOOM_REUSABLE_TIME_AUTHORITY_WITNESS`; arbitrary signer
  executables and caller-selected declarations are not accepted. Every terminal or interrupted
  attempt generation receives a second authority signature
  over its status, cumulative elapsed time, generation, predecessor hash, and durable journal hash.
  The command first writes and fsyncs an elapsed-neutral terminal journal, then asks the authority to
  issue an elapsed seal. Final artifact verification, output-path validation, report serialization,
  atomic replacement, file and directory fsync, and the seal made durable inside that report are
  also recurring work. The writer therefore measures an initial durable write, requests a
  conservative future-charged seal, and repeats only when the signed elapsed floor did not cover the
  completed write. Thus the final durable report contains a signed elapsed value no lower than its
  actual terminal boundary without requiring a journal or report to contain its own digest.
  The packaged authority maintains a signed append-only event chain plus an independently keyed,
  separately stored monotonic witness. The witness directory must live on a durability boundary
  that operators cannot roll back with the authority state snapshot; formal evidence is not honest
  if both directories can be restored together. It verifies registered starts, monotonic journal
  generations and elapsed floors,
  chronological bootstrap creation/reuse, latest-head continuity, ledger rollback, and authority
  identity resets. Ledger, witness, and authority-head transitions are interruption-recoverable: a
  signed ledger intent is completed deterministically after a stop at either later durable boundary.
  `benchmark_protocol.time_authority` discloses and binds the authority key, witness key, witness
  identity, creation time, and witness location into the run identity.
  Signing private keys and ledger state never enter the manifest, report, or
  benchmark artifact directory. Fixture signing remains pinned separately and claim-ineligible.
- `parity_certificate`: the current certificate availability and, when unavailable, its reason.
  An unavailable certificate is recorded as a claim-ineligibility reason but does not prevent a
  development run.
- `wad_profile`: required for non-fixture runs and validated through the immutable
  `freedoom2-deathmatch-v1` binding before training. The trainer and evaluator must report the same
  IWAD and PWAD hashes. CPU-only fixture manifests may omit real WAD assets.

The command starts the reusable timer before argument parsing, manifest and configuration
validation, identity and input hashing, artifact setup, and continuation or recovery verification.
That recurring command-setup duration is included independently in every active seed's elapsed
outcome; it is retained once as cohort activity and is not summed into a second aggregate duration.
For each seed it invokes the existing
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

An interrupted or crashed trainer may leave a recovery checkpoint only when its metrics prove that policy,
optimizer, RNG, and progress state are all restorable and the checkpoint carries the exact run and
attempt identities. The report retains a hashed recovery journal binding that checkpoint, progress,
and accumulated reusable elapsed time. A later `--merge` resumes the same attempt, adds new elapsed
time to the journaled elapsed time, and preserves the original cold-start identity. Completed and
failed seeds are reused unchanged and are never rerun or replaced. Every attempt also has a durable,
never-overwritten state-journal generation. Local hashes detect ordinary damage; the independently
signed journal head prevents an operator from erasing time by rewriting all local JSON and
recomputing those public hashes. A merge must name the latest on-disk chained generation; formal
continuation also verifies that head against the external monotonic witness. If interruption occurs
after a final same-head reseal but before report replacement, the preceding durable report remains
recoverable: `--merge` may upgrade it only to the authority's latest attestation for the exact same
generation, predecessor, journal hash, and status. A different generation or journal remains stale
and is rejected.

Before every trainer launch, the command writes and periodically refreshes a durable, checksummed
live-attempt journal. Training and 100-episode evaluation subprocesses both receive forwarded
SIGINT/SIGTERM only after that journal is refreshed. If the public command itself is interrupted
before it can write a report, a later invocation with the identical manifest discovers whether the
durable phase was training or evaluation. It verifies every bound identity and artifact, resumes
training state when needed, or reruns only the interrupted evaluator against the already-durable
checkpoint without replacing the attempt. A crash that produces no complete recovery checkpoint
remains terminal.
The real environment exposes a deterministic live-snapshot codec. Evidence checkpoints retain and
restore every direct environment and engine tensor, host reset RNG state, current observation and
context, episode start/done flags, in-progress returns and lengths, stable lane identities, reward
shaper state, policy, optimizer, AMP GradScaler state, original encoder-anchor targets, and
Python/NumPy/Torch/CUDA RNG states. Live restoration requires the exact saved lane count; an ordinary
non-evidence resume with a different environment count uses the documented lane-migration reset
path. Resume validation requires that complete inventory before the public command may advertise
the interruption as restorable. Ordinary legacy FP32 checkpoints may omit the disabled GradScaler
state; mixed-precision evidence checkpoints remain required to carry it.

`benchmark_protocol.continuation_identity` separately hashes the exact schema/trainer contract,
recipe, assets and bootstrap bytes, training/evaluation seeds, and timer phases/boundaries. Any
recipe, asset, seed, timer, schema, code provenance, WAD binding, evidence index, or artifact mismatch
fails before additional training work begins.

### Reusable-time authority operations

Provision authority state and its independently retained monotonic witness once. The witness must
not be included in snapshots or rollback domains containing the authority state:

```bash
gradoom-time-authority \
  --state-directory /secure/gradoom-time \
  --witness-directory /independent-append-only/gradoom-time-witness \
  init
export GRADOOM_REUSABLE_TIME_AUTHORITY_STATE=/secure/gradoom-time
export GRADOOM_REUSABLE_TIME_AUTHORITY_WITNESS=/independent-append-only/gradoom-time-witness
```

`start-attempt` reads `{"seed": 123}` on standard input and returns the signed anchor used in the
manifest. Bootstrap setup uses `create-bootstrap`, then a later invocation of
`record-bootstrap-reuse`, and finally `attest-bootstrap-reuse`; each accepts and emits canonical JSON
on standard input. The evidence command calls the same installed implementation directly for
journal sealing and latest-head/bootstrap verification. Copying an old ledger or coherently restoring
the whole authority-state directory under a newer witness is detected as rollback. The witness also
prevents silently reinitializing only the state directory. Deleting both trust domains creates a
different public identity, so old anchors and attestations are rejected, but doing so cannot support
a continuity claim.

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
