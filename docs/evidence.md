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
including recurring process initialization, uncached or per-process compilation, warm-up, training,
checkpoint evaluation, and durable checkpoint writing in measured elapsed time.
The standalone trainer rounds an unaligned request down to the preceding complete-rollout boundary,
never up. A benchmark segment is accepted only when that execution boundary equals its exact
predeclared checkpoint; otherwise it fails closed without evaluation. The final report output must
not alias any generated checkpoint, seed file, or metrics file through an equal path, symlink, or
hardlink.

Evaluation uses the same trainer executable with `--evaluate-checkpoint`,
`--evaluation-episodes 100`, `--evaluation-seeds-file`, and stochastic actions. Every evaluator must
return all 100 episode records in the predeclared order. Each record must have a positive episode
length and exactly one true terminal flag (`terminated` or `truncated`); zero-length, unterminated,
and contradictory records are evidence failures and cannot count toward the cohort. Passage is
computed solely from the completed episodes' mean `player_killcount` and requires at least `30.0`;
separately reported `compatibility_killcount` is a diagnostic and cannot change the verdict. The
passing checkpoint, metrics, all preceding checkpoint outcomes, all episode outcomes, and every
process or evidence failure are retained and hashed in the report evidence index.

The report has `authoritative: false` and `claim_eligible: false` unconditionally. Its per-seed
attempt states are `succeeded`, `exhausted`, `crashed`, `evaluation_failed`, or `evidence_failed`.
The workflow status is `passed` only when every predeclared development attempt succeeds; either
status remains ineligible for public claims. Benchmark continuation and bootstrap exclusions are not
part of this workflow version; `--merge` is rejected for development benchmarks.

The standalone trainer's `--evaluation-seeds-file` accepts a UTF-8 JSON array of exactly 100 unique
uint32 game seeds. Its emitted config binds the complete ordered seed list, file hash, stochastic
action setting, `player_killcount` gate signal, and separate compatibility signal into the retained
evaluation evidence.

Every development report includes `diagnostics.fixed_time`. When the separate diagnostic has not
been run, that field has `status: unavailable`, an explicit reason, and `affects_passage: false`.
The report also marks `public_performance_evidence.complete: false`; omission is allowed for cheap
development iteration but cannot be silently promoted into a complete public evidence bundle.

## Fixed-time training diagnostic

The `fixed_time_training_diagnostic` workflow compares final policy quality at one common reusable
wall-clock budget without extending or changing the matched benchmark's time-to-threshold run. Its
manifest uses the common envelope fields and a `diagnostic` object containing:

- `reusable_time_budget_seconds`: one finite positive budget used by every predeclared seed;
- `training_seeds`, `evaluation_episode_seeds`, and `evaluation_action_seed`: the same ordered seed
  declarations as the matched benchmark; the held-out grid contains exactly 100 unique uint32 game
  seeds;
- `recipe`: the exact standalone trainer `command` and fixed `arguments` used by the benchmark;
- `timing_rules`: the fixed reusable-run boundary: an outer monotonic wall clock starts before
  attempt setup and public subprocess launch and stops after trainer exit and durable training
  evidence writes. Seed-manifest creation, interpreter/module startup, recurring initialization,
  per-process or uncached compilation, warm-up, training, checkpoint writing, and training-metrics
  writing are included, with device synchronization around measured GPU work. Only the final
  held-out evaluation is outside the fixed training budget;
- `matching_benchmark_report`: the path and SHA-256 of the existing development or primary report;
- `artifacts_directory`: the immutable run-identity directory for seed grids, final checkpoints,
  training metrics, and evaluation metrics; and
- `wad_profile`: required for non-fixture runs and required to have the same binding identity as the
  matched benchmark.

Before starting training, the command verifies the matched report's digest, evidence index, run
identity, evidence level, fixture status, code provenance, recipe, training seeds, evaluation seeds,
action seed, and WAD-profile binding. Any mismatch fails before creating diagnostic artifacts. The
trainer starts every diagnostic seed from fresh policy and optimizer state and receives the outer
absolute monotonic deadline. It starts no rollout after that deadline; a deadline consumed by
startup therefore produces zero training transitions, while an in-flight rollout may finish before
the final checkpoint and training evidence are durably written. The command then evaluates that
checkpoint through the same stochastic 100-episode GraDOOM evaluator used by the benchmark.

Completed attempts retain exactly 100 genuinely completed episode records and report
`final_mean_player_killcount`. Throughput is computed as transitions per second and simulated tics
per second using the actual outer public-command elapsed time, never a trainer-reported inner timer;
the report retains transition count, frame skip, simulated tic count, elapsed time, timer source,
and the complete workload boundary. Compatibility `killcount` may be retained as a diagnostic but
is not final policy quality.

Declared input names must not collide with manifest, matched-report, or generated-artifact evidence
names. After every attempt, the command rehashes each generated seed manifest, checkpoint, training
metrics file, and evaluation metrics file and reconciles it with one uniquely named evidence-index
entry. The report retains both digests and a matched/mismatched status; a missing, ambiguous, or
changed binding is retained as an `artifact_evidence` failure and makes the attempt
`evidence_failed`.

All fixed-time results live under `diagnostics.fixed_time`, are non-authoritative on their own, have
`affects_passage: false`, and repeat the matched benchmark passage status as `unchanged: true`.
Process and evidence failures remain explicit failed attempts. A complete public performance bundle
requires a completed, matching non-fixture diagnostic and a separately claim-eligible benchmark;
development and fixture reports remain incomplete regardless of diagnostic quality or throughput.
`--merge` is not supported for this workflow version.

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
