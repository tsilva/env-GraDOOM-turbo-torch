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
  `-c`, and statically visible opaque process indirection are rejected. Relative executables and
  scripts are resolved from the manifest directory. The entry point, its transitive imports, and
  every regular file recursively below the declared code root are hashed into the recipe identity
  and freshly re-inventoried before the final durable report. Verification requires exact
  relative-path membership, bytes, regular-file identity, mode, and size; additions, removals,
  replacements, and type changes fail closed. Each initial file object remains pinned by a private,
  non-inheritable read-only descriptor through final verification, so unlink-and-recreate is detected
  even on filesystems that immediately reuse inode numbers. The boundary is limited to 4,096 files
  and is also preflighted against the process soft open-file limit, its current open descriptors, and
  a 64-descriptor runtime reserve; insufficient capacity and mid-inventory exhaustion fail closed,
  and all pins are released on success or error. File and directory symlinks are rejected before any
  target resolution, whether their targets are internal or external. No directory name, suffix,
  executable bit, UTF decoder, parse probe, or compression format exempts bytes, because Python can
  decode or decompress arbitrary local payloads before computed execution. Manifest and merge-report
  files inside the root are ordinary bound members rather than exclusions. The benchmark artifact
  root must be disjoint from the code root, and the public report must be outside the code root.
  Before the first trainer subprocess, those exact bytes and every Python source reachable from the
  selected interpreter environment are streamed into a deterministic kernel-sealed in-memory ZIP.
  Only native Python extensions reachable from the trainer's sealed import closure and their
  resolved transitive ELF dependencies are copied into individually sealed memory files; an unused
  optional extension with unavailable libraries cannot invalidate an unrelated recipe. The Python
  executable and its exact ELF interpreter are sealed as well, and the child is launched by that
  inherited interpreter descriptor rather than the mutable filesystem loader. Absolute
  `DT_NEEDED` entries that cannot be redirected to a sealed descriptor fail closed. Exact loader
  names, resolved paths, dependency bytes, and effective library-search inputs are hashed into the
  environment identity.
  Each child resolves dependency SONAMEs through a private read-only directory whose entries point
  only to the inherited sealed descriptors, so changing or removing a mutable library after binding
  cannot change executed bytes. Aggregate environment identity, archive hash, and bootstrap-loader
  hash are part of the recipe identity, so
  changing an external package between an interrupted attempt and its continuation is an unlike-run
  failure. The entry point, local imports, external Python packages, and source introspection read
  only those inherited sealed bytes for every training and evaluation child; mutable code-root and
  environment paths are removed from the child import search path. Generated compilation is
  accepted only after a blocking, authenticated anonymous channel asks the parent to authorize the
  exact source or marshalled code-object identity. The child cannot continue from the audit event
  until the parent has durably refreshed the live interruption journal. Once an attempt is resumed,
  that generated-semantic set is frozen: identical repeated use remains valid, while a new or
  changed external-derived payload fails before execution instead of being appended. Generated-code
  records are retained in the attempt-level execution-recipe hash and authority-signed journal, so
  append-safe reuse binds actual generated bytes rather than trusting a caller filename or mutable
  stderr transcript. The seal is bounded to
  16,384 payloads, 16 MiB per source payload, 512 MiB per native extension, 1 GiB per native shared
  library, and 2 GiB in aggregate, with
  excess rejected before the payload is read. A child audit hook rejects relative writes and writes,
  additions, removals, replacements, links, and metadata mutations anywhere under the protected
  roots, including `dir_fd` attempts and attempts that would restore the original bytes before final
  verification. It also rejects protected mutable-source reads, unbound subprocess/native-library
  execution, process replacement, opaque code construction, and dynamic compilation or execution
  not attributable to sealed bytes. Trainer
  subprocesses cannot write Python bytecode into a protected source boundary. This full-lifetime
  binding is independent of how code is reached, so computed builtins, reflection, plugin discovery,
  namespace aliases, and indirect process replacement cannot execute mutable Python bytes.
- `artifacts_directory`: the directory under which the run-identity and per-seed artifacts are
  durably retained. It may neither contain nor be contained by `trainer.code_root`, preventing
  growing benchmark outputs from becoming executable aliases or self-invalidating recipe members.
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
  issue an elapsed seal. Before another seed can execute or the public report can be written, the
  signed attempt and its exact run, recipe, seed, generation, journal, evidence-index, and artifact
  provenance are atomically replaced and file/directory-fsynced in a per-generation attempt seal.
  An identical-manifest invocation discovers and verifies these seals, recovers an authority's
  newer same-head attestation when necessary, and never reruns a sealed seed merely because the
  public report was lost. Final artifact verification, output-path validation, report serialization,
  atomic replacement, file and directory fsync, and the seal made durable inside that report are
  also recurring work. The writer therefore measures an initial durable write, requests a
  conservative future-charged seal, and repeats only when the signed elapsed floor did not cover the
  completed write. Thus the final durable report contains a signed elapsed value no lower than its
  actual terminal boundary without requiring a journal or report to contain its own digest.
  Pre-command anchors establish attempt identity and append continuity; they are not treated as one
  continuously running stopwatch across the cohort. Each seed accumulates the common recurring
  setup/finalization costs, only that seed's active training/evaluation and recovery spans, and only
  its own sealing overhead. Work performed for another seed is never added merely because its anchor
  was issued earlier. Before argument parsing, the authority opens one invocation-setup segment;
  after manifest validation, environment sealing and hashing, artifact setup, and recovery
  verification, it seals that shared duration against every active attempt. Immediately before each
  active seed begins, it then opens a seed-local segment. Journal sealing adds the authority-owned
  setup duration to only that seed's active interval. Both measurements use the same-boot monotonic
  clock exclusively; a reboot, unavailable boot identity, or attempted wall-clock adjustment fails
  closed instead of supplying an operator-adjustable fallback. Every journal seal consumes exactly
  one seed-local segment and enforces its independently observed duration in addition to the
  caller's conservative floor, so caller under-reporting cannot shorten active work without charging
  idle anchor age or another seed's execution.
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
- `cuda_residency_acceptance`: an optional boolean, defaulting to `false`. When true, the evidence
  command owns the standalone trainer's `--cuda-residency-acceptance` flag and requires a passing
  `gradoom-cuda-residency-v1` record from every training segment. Trainer arguments cannot override
  this setting.

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
and is rejected. If no public report became durable, the same exact-head recovery instead starts
from the per-generation signed attempt seal and applies the same provenance and authority checks.

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
journal sealing, seed-local timing-segment activation, and latest-head/bootstrap verification. The
low-level `start-invocation`, `seal-invocation-setup`, and `start-timing-segment` operations expose
the same authority-owned setup and seed-local boundaries for audit fixtures; the public evidence
command invokes them itself, so operators do not add them to manifests. Copying
an old ledger or coherently restoring
the whole authority-state directory under a newer witness is detected as rollback. The witness also
prevents silently reinitializing only the state directory. Deleting both trust domains creates a
different public identity, so old anchors and attestations are rejected, but doing so cannot support
a continuity claim.

The standalone trainer's `--evaluation-seeds-file` accepts a UTF-8 JSON array of exactly 100 unique
uint32 game seeds. Its emitted config binds the complete ordered seed list, file hash, stochastic
action setting, `player_killcount` gate signal, and separate compatibility signal into the retained
evaluation evidence.

### CUDA residency acceptance

CUDA residency acceptance instruments the real standalone trainer used by the development
benchmark (and available to the primary benchmark through the same trainer contract). It begins
only after `--steady-state-after-rollouts` and fails closed if no later rollout is checked. The
record verifies one concrete CUDA device for observations, actions, rewards, reset selectors and
state, rollout state, inference outputs, loss tensors, optimizer state, parameters, and gradients
used for updates. One continuous rollout scope covers observation augmentation and staging,
transitions, reward shaping, rollout writes, context/reset updates, value bootstrap, and rollout
finalization; one continuous update scope covers PPO staging, losses, and parameter updates. Both
reject host-to-accelerator and accelerator-to-host copies before they execute, including
`.cpu().numpy()` round trips and NumPy-created update tensors.

The retained record names the exact checked workload and transition count; GPU model, concrete
device, compute capability, and memory; and Python, GraDOOM, Torch, CUDA, cuDNN, and NumPy versions.
Its host-transition guard count and zero-transfer result are part of the validated evidence. A
fixture process may exercise this report contract only as `fixture_contract`; fixture reports remain
non-authoritative and claim-ineligible and cannot impersonate CUDA hardware evidence.

Acceptance guards exclude bounded scalar telemetry, configuration and scheduling, process
bootstrap, and checkpoint extraction/writing. Those operations remain permitted outside the
steady-state data plane. When the option is disabled, the trainer creates no acceptance collector,
uses two reusable no-op contexts per rollout instead of per-step or per-minibatch branches, emits no
acceptance record, and performs no additional host transport.
Hardware integration tests require an explicitly allocated CUDA device and
`GRADOOM_RUN_CUDA_ACCEPTANCE=1`; otherwise they skip with a clear reason while deterministic
contract and fixture-wiring tests continue to run.

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
- `invariant_suite` (optional until a provider runtime is available): version `1.0.0`, a `mode` of
  `fixture` or `real`, the `runner_input` name of the hash-verified repository runner, and optional
  statuses for the retained deep diagnostics. Manifests cannot declare provider commands. Fixture
  mode requires `fixture: true`; its executable examples use `pass`, while the
  `reward_mismatch`, `missing_player_killcount`, and `missing_termination` cases are retained
  adversarial test inputs.
- `invariant_suite.real_configuration` (required in real mode): a requested `device` (`cpu`,
  `cuda`, or canonical ASCII `cuda:N`), a predeclared `timeout_seconds`, a
  `reference_scenario_config_input` naming a hashed declared input, and `semantic_probes` for
  `termination`, `truncation`, `player_killcount`, and
  `player_killcount.enemy_on_enemy_exclusion`. Each probe declares exactly two uint32 `seeds`, a
  non-empty cycle of two-lane pinned action-index rows under `actions`, and a positive
  `max_steps` no greater than 100,000. The timeout must be at least 120 seconds and at least 0.05
  seconds per total declared probe step, and cannot exceed 86,400 seconds. The action rows retain
  their published action meanings; success comes only from an event observed in public step
  results.
- `wad_profile` (required when `certified_freedoom2_wad_profile` is declared available): the
  `freedoom2-deathmatch-v1` profile ID and exactly one `gradoom` and one `env-vizdoom-turbo`
  provider binding. Each binding declares `iwad_path`, `pwad_path`, and the complete policy-facing
  `configuration`. Relative WAD paths are resolved from the manifest directory.

Real invariant execution additionally requires that `wad_profile` validation has matched. The
command derives both providers' IWAD/PWAD paths, hashes, map, skill, scenario, action mode, frame
skip, horizon, and preprocessing directly from that validated binding rather than accepting a
second copy in `real_configuration`. The reference scenario config must sit beside, and therefore
load, the exact validated reference `deathmatch.wad`; a substitute PWAD fails before provider work.

The reference config is an exact allowlisted part of the shared scenario configuration. It must
declare only the bound PWAD, skill, resolution, HUD and screen-flash settings, episode start and
timeout, player mode, the complete pinned button set, and the native `HEALTH`, `KILLCOUNT`, and
`PLAYER_KILLCOUNT` variables. Extra provider-only behavior settings, missing settings, duplicate
settings, or different values fail before provider construction. `episode_return` remains a derived
report signal selected through `info_filter`; it is never passed as a native game variable.

Real kill probes use an additional diagnostic-only actor stage under the same bound IWAD, PWAD,
map, skill, action table, and provider configuration. The player-kill stage contains exactly the
controlled player and one enemy. The infighting stage contains exactly the controlled player and
two enemies; a harmless west-facing shot during setup wakes the east-side actors without damaging
them. GraDOOM records source and target actor IDs where engine damage becomes a death. The pinned
reference provider enables native object information before initialization and proves the same
death from stable object IDs and the distinct surviving actor in the isolated stage. A passing
event must have exactly one death, retain the independently observed attacker, remove the target,
and contain no additional actors. Counters, rewards, requested actions, provider labels, and pixel
changes never supply actor identity. Missing object support, changed assets, replayed stages,
self-attribution, multiple sources, or ambiguous populations produce a named failed invariant.
This instrumentation is inactive during ordinary reset, step, policy evaluation, and benchmarks.

The repository fixture is the executable example of this schema:
[`tests/fixtures/evidence/readiness-manifest.json`](../tests/fixtures/evidence/readiness-manifest.json).
The independently versioned two-provider invariant example is
[`tests/fixtures/evidence/invariant-readiness-manifest.json`](../tests/fixtures/evidence/invariant-readiness-manifest.json).

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

## Fast Turbo invariant suite

Invariant suite `1.0.0` runs before readiness is decided. The manifest cannot supply provider
commands. It names the repository-owned invariant runner as a declared input; the command verifies
that input's path and hash, invokes only the installed runner module, and authenticates the response
with a fresh challenge plus the runner-source digest. Arbitrary executables and static contract
emitters therefore cannot satisfy the suite.

The runner exercises each provider through public construction, reset, step, and masked-reset
operations. The command requires the complete common constructor signature and defaults, full
action meanings, exact observation, signal, and reward shapes and dtypes, lifecycle operations,
termination, truncation, manual episode-reset semantics, and `player_killcount`. It additionally
requires every GraDOOM reset-mask and action input plus reset and step output to be a Torch tensor on
the declared device. A terminal event passes only after the terminal lanes reset and public stepping
resumes. Masked reset is checked both at its immediate return and on the following transition against
deterministic one-step and two-step controls, proving selected-lane reset and unselected-lane
continuation in provider state. The player-attributed kill probes require staged actor/target
attribution in addition to counters: the former must observe a player-to-enemy event and increment
`player_killcount`, while the latter must observe an enemy-to-enemy event and increment only
compatibility `killcount`. For real providers, the repository-owned oracle rehashes the validated
IWAD and PWAD at the event boundary and consumes a freshly staged actor-population token exactly
once; any public reset invalidates that stage. GraDOOM uses engine-recorded source and target actor
IDs from the damage/death site. The reference provider instead uses stable native object IDs and the
isolated before/after population to identify the distinct surviving attacker and removed target. The
suite reconciles the exact staged and surviving populations and exactly one death with the public
counter deltas: player-to-enemy increments `player_killcount`, while enemy-to-enemy leaves it
unchanged and increments compatibility `killcount`. Requested actions, counters, rewards,
observations, and pixels never supply actor identity. Provider- or manifest-supplied attribution,
reset or stage replay, ambiguous populations, and counter-only evidence fail closed.

Real execution loads GraDOOM and the immutable reference revision independently through the pinned
reference adapter. An absent optional provider runtime is unavailable; changed assets, an invalid
binding, a non-exact scenario config, or an unprovable executed GraDOOM Git checkout fail closed.
The GraDOOM revision is derived from the clean checkout containing the executed module rather than
copied from manifest provenance. Once the runtime is present, missing signals,
malformed transitions, runtime errors, and unobserved lifecycle or kill events become named failed
invariants rather than generic unavailability. Non-fixture execution also requires the GraDOOM
provider revision to match `code_provenance.revision`. The fixture runner provides deterministic
public-operation probes only when the manifest itself is `fixture: true`; fixture evidence can never
support a real claim.

Real runner timeout uses the accepted predeclared timeout rather than an unconditional process
timeout. Exhaustion is retained as a named unavailable result and cannot produce readiness or a
claim; incoherent timeout/probe budgets are rejected before execution.

The report records every check under `invariant_suite.checks`. A mismatch sets both the suite and
readiness status to `failed` and names the public `behavior`; missing, unconfigured, or unavailable
suite execution leaves readiness `unavailable`. A complete invariant pass still reports
certification unavailable when the required real pretrained policy corpus is unavailable or was not
declared, with `claim_eligible: false`.

Mechanics, trace, outcome-distribution, policy-observation, and rendering diagnostics remain
separate reproducible tools. Their declared statuses are copied under
`invariant_suite.diagnostics` with `affects_verdict: false`; they never change the invariant or
readiness verdict.

## Report schema version 1

The JSON report records its schema version, workflow, evidence level, fixture state, readiness
status, claim eligibility and structured reasons, stable run identity, declared code provenance,
declared inputs, prerequisites, optional WAD-profile validation, and evidence index. A fixture
report is `unavailable` while real prerequisites are missing, has `claim_eligible: false`, and
names every missing prerequisite in `claim_reasons`. Every report records the invariant-suite
version even when provider execution has not yet been configured.

`run_identity` is the lowercase SHA-256 digest of canonical JSON containing the manifest schema,
workflow, evidence level, fixture state, code provenance, input names and declared hashes,
prerequisite identifiers, and—when supplied—the complete normalized WAD-profile binding. Input and
prerequisite ordering does not affect it. A configured invariant suite also binds its independent
version and each provider's revision and canonical contract digest. Canonical JSON uses
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
