# Architecture

## Performance boundary

The certified training path is a device-resident state machine. Host code may configure a run, compile scenario assets, launch work, and consume bounded telemetry; it must not participate in each environment step. Reset/step transitions and state indices use Torch tensors exclusively; only diagnostic RGB rendering crosses through NumPy.

## Layers

1. The scenario compiler reads an external IWAD/PWAD and produces immutable structure-of-arrays geometry and actor metadata.
2. The device engine owns batched match state, deterministic counter-based random state, mechanics, collision, combat, rewards, resets, and observations.
3. `GraDoomVecEnv` exposes the supported `vizdoom-turbo` surface plus a device-tensor contract.
4. GradLab consumes device tensors directly and evaluates checkpoints in unmodified ViZDoom.

The first implementation uses vectorized Torch operations so the same code is testable on CPU and CUDA. Profiling determines which operations become fused C++/CUDA kernels. The public contract and scenario representation do not depend on the kernel implementation.

## Certified profiles

- `deathmatch-p1-v1`: one independently controlled player per match, the pinned ViZDoom deathmatch scenario, 17 discrete actions, frame skip 2, 84x84 grayscale CHW stack 4, player-death termination, and 4200-tic truncation.
- Future multiplayer and multi-WAD profiles compile independently. They may share implementation, but they may not add material overhead to `deathmatch-p1-v1`.
