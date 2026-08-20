## PROJECT PURPOSE

`env-Doom-turbo-torch` is a Torch-native Doom reinforcement-learning environment whose primary outcome is the lowest reproducible wall-clock time required to train policies that perform zero-shot in comparable reference ViZDoom environments.

## PROJECT REQUIREMENTS

### Performance

- `env-Doom-turbo-torch` must pursue the fastest reproducibly benchmarked Doom training throughput in the world, with time-to-reference-target as its primary performance metric.
- Simulation, observations, actions, rewards, resets, rollouts, policy inference, and learning must remain GPU-resident during steady-state training.
- The initial certified hardware target is one NVIDIA RTX 4090 integrated with GradLab.
- Generality and future features must not impose material overhead on certified fast paths.
- Any public fastest claim must use current, reproducible, workload-equivalent benchmark evidence.

### Transfer and parity

- `env-Doom-turbo-torch` must preserve Doom gameplay mechanics sufficiently for policies trained in it to remain performant without fine-tuning in comparable ViZDoom environments.
- Deterministic gameplay mechanics must match the reference environment unless an explicitly documented deviation passes transfer acceptance.
- Minor stochastic divergence is acceptable when its distributions remain compatible and it does not materially harm policy transfer.
- Raw fidelity evaluation must precede observation preprocessing and use ViZDoom deathmatch’s 320×240 RGB24 output with the full Doom HUD.
- Pixel-exact rendering is not required, but policy-facing observations must support reliable transfer.
- Domain randomization may improve resilience to bounded rendering and simulation differences but must not conceal material semantic incompatibility.

### Compatibility and content

- Use `env-Doom-turbo-torch` as the project and GitHub repository name, `env-doom-turbo-torch` as the Python distribution name, and `env_doom_turbo_torch` as the public Python import package; current project content must not use any former project name or import identifier.
- `env-Doom-turbo-torch` must provide the supported deathmatch API exposed by `env-ViZDoom-turbo`.
- Its Turbo-compatible reset and step data plane must accept and return Torch tensors; it must not provide a NumPy transition transport.
- The first certified environment is the ViZDoom deathmatch scenario.
- Certified environments must support user-supplied Doom II and Freedoom WADs.
- Future releases must support multiple certified deathmatch WADs without slowing the initial single-scenario fast path.

### Multiplayer

- Future releases must support multi-agent training with multiple independently controlled players.
- Future multiplayer must allow matches containing configurable combinations of trainable policies, frozen policies, humans, scripted opponents, and monsters.

### Licensing

- License permissiveness is subordinate to training throughput, semantic parity, and policy transfer.
- `env-Doom-turbo-torch` may reuse compatible reference-engine source when doing so advances the project objective and all provenance and redistribution obligations are satisfied.
