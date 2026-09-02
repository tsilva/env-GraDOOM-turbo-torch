# Changelog

## Unreleased

- Add a matching fixed-time training diagnostic to the public evidence command, with exact
  100-episode player quality, end-to-end simulated-tic and transition throughput, and explicit
  separation from benchmark passage. The reusable-time boundary includes public-process startup,
  and generated artifacts are digest-bound to unique retained evidence entries.
- Rename the project and distribution to `env-GraDOOM-turbo-torch` and
  `env-gradoom-turbo-torch` while restoring the published `gradoom`,
  `GraDoomVecEnv`, and `GraDOOM-v0` public API names.
- Keep package imports and CPU execution available when the CUDA-only Triton
  runtime is not installed.
- Migrate `GraDoomVecEnv` to the breaking Turbo Vector API v2 shared
  constructor and defaults while keeping reset and step transitions Torch-only
  on `env.device`.
- Add immutable exact capabilities, portable signal schemas, numeric reset
  infos, deterministic catalog index zero, standardized async stepping, and
  opt-in RGB rendering.
- Reject non-neutral `num_threads` because the device path has no host worker
  pool.
- Preserve the certified deathmatch training, playback, CUDA smoke, and CUDA
  benchmark profiles through explicit construction settings.
- Point reference-provider documentation, diagnostics, and local asset defaults
  at the standardized `env-ViZDoom-turbo` project and `env-vizdoom-turbo`
  distribution names.
- Add the independently versioned fast Turbo invariant suite to parity readiness,
  including named public-behavior failures and GraDOOM Torch device checks.
- Add sealed two-origin pretrained-policy corpus validation and exhaustive,
  restart-safe 256-seed execution in both providers through the public evidence
  command, with deterministic nonclaiming subprocess fixtures.
- Seal the policy runner and policy bytes used by corpus execution, revalidate
  resumed outcomes canonically, and durably checkpoint each completed batch.
- Close corpus-resume and schema gaps by rejecting partial batches, unsupported
  model contracts, undeclared override fields, tampered provenance, and unsafe
  JSON numeric values through the public evidence command.
