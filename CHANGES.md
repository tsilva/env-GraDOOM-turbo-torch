# Changelog

## Unreleased

- Migrate `EnvDoomTurboTorchVecEnv` to the breaking Turbo Vector API v2 shared
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
