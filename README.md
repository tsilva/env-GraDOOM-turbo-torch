<p align="center">
  <img src="https://raw.githubusercontent.com/tsilva/env-Doom-turbo-torch/main/logo.png" alt="env-Doom-turbo-torch" width="560" />
  <br />
  <strong>🔥 Train Stronger Doom Policies, Faster 🔥</strong>
</p>

`env-Doom-turbo-torch` is a Python library and integrated training system for expert reinforcement-learning researchers who want to train strong Doom deathmatch policies from fresh initialization on NVIDIA GPUs. A certified result counts only when the unchanged stochastic policy transfers to `env-ViZDoom-turbo`; certification ranks policy quality by systematic player-attributed kills, with reusable-run wall-clock time and raw simulated Doom tics breaking close ties.

Its batched simulation, rendering, rewards, resets, rollouts, policy inference, and learning remain in Torch on the GPU during steady-state training. Use `EnvDoomTurboTorchVecEnv` with an operator-supplied Doom II or Freedoom IWAD and the pinned ViZDoom deathmatch scenario.

## Install

`env-Doom-turbo-torch` requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv add env-doom-turbo-torch
```

For source development:

```bash
git clone https://github.com/tsilva/env-Doom-turbo-torch.git
cd env-Doom-turbo-torch
uv sync --group dev
```

## Use

```python
import gymnasium as gym
import torch

num_envs = 128
device = torch.device("cuda")
env = gym.make_vec(
    "env_doom_turbo_torch:EnvDoomTurboTorch-v0",
    game="VizdoomDeathmatch-v1",
    scenario="/path/to/vizdoom/scenarios/deathmatch.wad",
    rom_path="/path/to/doom2.wad",
    num_envs=num_envs,
    device=device,
    render_mode="rgb_array",
    compile_engine=True,
)

lanes = torch.arange(num_envs, device=device)
observations, signals = env.reset_device(
    torch.ones(num_envs, device=device, dtype=torch.bool),
    lanes + 1,
)
actions = lanes % env.single_action_space.n
transition = env.step_and_reset_device(actions, lanes + num_envs + 1)
raw_rgb_with_hud = env.render()  # 320x240 RGB24, no observation preprocessing
env.close()
```

The module-qualified ID imports the package and registers the factory. This ID
is vector-only, requires an explicit `game`, and returns the native
Torch-only `EnvDoomTurboTorchVecEnv`; the class also remains available for direct use.

`observations`, rewards, episode flags, and signals remain Torch tensors on the selected device.
Request the env-Doom-turbo-torch-specific `player_killcount` game variable when policy quality
must count only enemy deaths delivered by the player. ViZDoom-compatible
`killcount` remains available and also includes countable monsters killed by
infighting in this single-player scenario.

## Commands

```bash
uv run pytest                                             # run the test suite
uv run ruff check .                                      # lint the repository
uv run python -m env_doom_turbo_torch.inspect_scenario \
  --scenario /path/to/deathmatch.wad --iwad /path/to/doom2.wad  # inspect assets
uv run python play.py --scenario /path/to/deathmatch.wad \
  --iwad /path/to/doom2.wad                              # play with keyboard controls
uv run python tools/cuda_correctness_smoke.py --compile-engine   # check CUDA residency
uv run python train.py --iwad /path/to/doom2.wad \
  --scenario /path/to/deathmatch.wad                    # standalone 256x16 PPO
uv run python train.py --iwad /path/to/doom2.wad \
  --scenario /path/to/deathmatch.wad --wandb            # log to GradLab's W&B project
uv run python train.py --initialize-from /path/to/policy.pt \
  --iwad /path/to/doom2.wad --scenario /path/to/deathmatch.wad  # warm-start lane only
uv run python tools/convert_gradlab_checkpoint.py \
  --source /path/to/published/model.zip \
  --output /path/to/standalone-policy.pt                # no GradLab/SB3 imports
uv run python tools/evaluate_vizdoom_checkpoint.py \
  --checkpoint /path/to/policy.pt --iwad /path/to/doom2.wad \
  --scenario-config /path/to/deathmatch.cfg                      # zero-shot transfer gate
```

## Notes

- `env-Doom-turbo-torch` is under active construction and is not yet parity-certified. No current release supports a public quality- or speed-leadership claim.
- Certification ranks results in this order: zero-shot transfer eligibility, systematic `player_kills`, reusable-run wall-clock time among practically equivalent policies, then raw simulated Doom tics.
- Certified results start from freshly initialized policy and optimizer state. Pretrained, adapted, fine-tuned, and warm-start runs are reported separately.
- Final certification uses five predeclared cold-start seeds, reports every outcome without replacement, and requires at least four unchanged stochastic policies to transfer equivalently to `env-ViZDoom-turbo`.
- The first certification candidate is single-player `deathmatch-p1-v1`: 17 actions, frame skip 2, and 84×84 grayscale CHW observations with four-frame stacking.
- `render()` and `render_lane()` expose the unprocessed 320×240 RGB24 comparison view with the full Doom HUD; observation preprocessing remains separate from this diagnostic render path.
- The initial certification hardware target is one NVIDIA RTX 4090 integrated with GradLab.
- Pass asset paths directly or set `ENV_DOOM_TURBO_TORCH_IWAD` and `ENV_DOOM_TURBO_TORCH_DEATHMATCH_WAD`. WADs and other game data are not distributed with this repository.
- Torch tensors are the only reset/step transition transport, including reset selectors and read-only state indices. Only diagnostic RGB arrays cross into NumPy.
- Operator-run benchmarks require a controlled quiet window and matched reference evidence; see [deathmatch parity](./docs/deathmatch-parity.md).
- The current internal RTX 4090 training optimization recipe and its three-seed evidence are recorded in [training optimization](./docs/training-optimization.md). These results are experimental and do not supersede the parity-certification requirement.
- See [third-party notices](./THIRD_PARTY_NOTICES.md) for source and game-data policy.

## Architecture

![env-Doom-turbo-torch architecture](https://raw.githubusercontent.com/tsilva/env-Doom-turbo-torch/main/architecture.png)

## License

The project's original source code is [MIT-licensed](./LICENSE). Bundled ZDoom BulletChip
resources retain their separate [GPL-3.0-only license](./LICENSES/GPL-3.0-only.txt); see
the [third-party notices](./THIRD_PARTY_NOTICES.md) for exact provenance and redistribution terms.
