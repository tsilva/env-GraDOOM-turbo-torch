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
return all 100 episode records in the predeclared order. Passage is computed solely from their mean
`player_killcount` and requires at least `30.0`; separately reported `compatibility_killcount` is a
diagnostic and cannot change the verdict. The passing checkpoint, metrics, all preceding checkpoint
outcomes, all episode outcomes, and every process or evidence failure are retained and hashed in the
report evidence index.

The report has `authoritative: false` and `claim_eligible: false` unconditionally. Its per-seed
attempt states are `succeeded`, `exhausted`, `crashed`, `evaluation_failed`, or `evidence_failed`.
The workflow status is `passed` only when every predeclared development attempt succeeds; either
status remains ineligible for public claims. Benchmark continuation and bootstrap exclusions are not
part of this workflow version; `--merge` is rejected for development benchmarks.

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
