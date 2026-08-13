<div align="center">
  <img src="./logo.png" alt="GraDOOM" width="560" />

  **🔥 Rip and Tear Until It Is Done—at GPU speed. 🔥**
</div>

GraDOOM is a Python library for reinforcement-learning researchers and engineers who need to train Doom policies at high throughput. It runs batched deathmatch simulation, rendering, rewards, and resets in PyTorch on the same device as the learner, then targets zero-shot evaluation in comparable ViZDoom environments.

Use `GraDoomVecEnv` with an operator-supplied Doom II or Freedoom IWAD and the pinned ViZDoom deathmatch scenario. The current alpha implements Turbo Vector API v2 with Torch-only transition tensors, alongside a synchronization-free device API, scenario compiler, and vectorized Torch execution model.

## Install

GraDOOM requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tsilva/GraDOOM.git
cd GraDOOM
uv sync --group dev
```

## Use

```python
import gymnasium as gym
import torch

num_envs = 128
device = torch.device("cuda")
env = gym.make_vec(
    "gradoom:GraDOOM-v0",
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
Torch-only `GraDoomVecEnv`; the class also remains available for direct use.

`observations`, rewards, episode flags, and signals remain Torch tensors on the selected device.

## Commands

```bash
uv run pytest                                             # run the test suite
uv run ruff check .                                      # lint the repository
uv run python -m gradoom.inspect_scenario \
  --scenario /path/to/deathmatch.wad --iwad /path/to/doom2.wad  # inspect assets
uv run python play.py --scenario /path/to/deathmatch.wad \
  --iwad /path/to/doom2.wad                              # play with keyboard controls
uv run python tools/cuda_correctness_smoke.py --compile-engine   # check CUDA residency
python train.py --iwad /path/to/doom2.wad \
  --scenario /path/to/deathmatch.wad                    # standalone 256x16 PPO
python train.py --iwad /path/to/doom2.wad \
  --scenario /path/to/deathmatch.wad --wandb            # log to GradLab's W&B project
python train.py --initialize-from /path/to/policy.pt \
  --iwad /path/to/doom2.wad --scenario /path/to/deathmatch.wad  # weights-only start
python tools/convert_gradlab_checkpoint.py \
  --source /path/to/published/model.zip \
  --output /path/to/standalone-policy.pt                # no GradLab/SB3 imports
python tools/evaluate_vizdoom_checkpoint.py \
  --checkpoint /path/to/policy.pt --iwad /path/to/doom2.wad \
  --scenario-config /path/to/deathmatch.cfg                      # zero-shot transfer gate
```

## Notes

- GraDOOM is under active construction and is not yet parity-certified. No current release supports a public fastest-training claim.
- The first certification candidate is single-player `deathmatch-p1-v1`: 17 actions, frame skip 2, and 84×84 grayscale CHW observations with four-frame stacking.
- `render()` and `render_lane()` expose the unprocessed 320×240 RGB24 comparison view with the full Doom HUD; observation preprocessing remains separate from this diagnostic render path.
- The initial certification hardware target is one NVIDIA RTX 4090 integrated with GradLab.
- Pass asset paths directly or set `GRADOOM_IWAD` and `GRADOOM_DEATHMATCH_WAD`. WADs and other game data are not distributed with this repository.
- Torch tensors are the only reset/step transition transport, including reset selectors and read-only state indices. Only diagnostic RGB arrays cross into NumPy.
- Operator-run benchmarks require a controlled quiet window and matched reference evidence; see [deathmatch parity](./docs/deathmatch-parity.md).
- The current internal RTX 4090 training optimization recipe and its three-seed evidence are recorded in [training optimization](./docs/training-optimization.md). These results are experimental and do not supersede the parity-certification requirement.
- See [third-party notices](./THIRD_PARTY_NOTICES.md) for source and game-data policy.

## Architecture

![GraDOOM architecture](./architecture.png)

## License

GraDOOM's original source code is [MIT-licensed](./LICENSE). Bundled ZDoom BulletChip
resources retain their separate [GPL-3.0-only license](./LICENSES/GPL-3.0-only.txt); see
the [third-party notices](./THIRD_PARTY_NOTICES.md) for exact provenance and redistribution terms.
