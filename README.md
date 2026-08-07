<p align="center">
  <img src="logo.png" alt="GraDOOM logo" width="720">
</p>

# GraDOOM

GraDOOM exists to produce the fastest reproducibly benchmarked Doom reinforcement-learning training environment on Earth. Its decisive metric is wall-clock time to train a policy that passes an unchanged evaluation in reference ViZDoom.

The first certified target is ViZDoom's single-player deathmatch scenario on one RTX 4090. The steady-state path keeps simulation, observations, actions, rewards, resets, rollout storage, policy inference, and learning on the GPU. Doom II and Freedoom data remain external inputs and are never committed to this repository.

## Status

GraDOOM is under active construction and is **not yet parity-certified**. The current implementation establishes the scenario compiler, device tensor contract, `vizdoom-turbo`-shaped vector API, and a vectorized reference execution model. A release may call itself certified only after its zero-shot policies pass the reference ViZDoom gate and its matched benchmarks pass the performance gate.

## Development setup

```bash
uv sync --group dev
uv run pytest
```

The initial deathmatch smoke can use the ViZDoom scenario WAD and either Doom II or Freedoom as the external IWAD:

```bash
uv run python -m gradoom.inspect_scenario \
  --scenario /path/to/vizdoom/scenarios/deathmatch.wad \
  --iwad /path/to/doom2.wad
```

## Performance discipline

Do not publish or infer a throughput claim from an unmatched run. The release benchmark must pin source and container digests, WAD hashes, action/observation/reward contracts, policy, hardware state, raw samples, and reference evaluation results. Beast-3 performance runs require an operator-confirmed quiet window.
