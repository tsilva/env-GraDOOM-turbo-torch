## PROJECT PURPOSE

`env-Doom-turbo-torch` is a Torch-native Doom reinforcement-learning environment and integrated training system built primarily for its creator to produce the strongest reproducible cold-start policies, measured by player-attributed kills, whose unchanged behavior transfers zero-shot to `env-ViZDoom-turbo`; it should also be independently reproducible and useful to expert practitioners. Among policies with practically equivalent quality, it minimizes reusable-run wall-clock training time and then raw simulated Doom tics.

## PROJECT REQUIREMENTS

### Objective and performance

- Certified results must begin with freshly initialized policy and optimizer state; pretrained, adapted, fine-tuned, and warm-start results belong to separate evidence lanes.
- Certification must prioritize zero-shot transfer eligibility, systematic `player_kills`, reusable-run wall-clock training time among practically equivalent policies, and raw simulated Doom tics in that order.
- End-to-end co-designed training results and fixed-stack environment comparisons must remain distinct evidence lanes.
- Simulation, observations, actions, rewards, resets, rollouts, policy inference, and learning must remain GPU-resident during steady-state training.
- The initial certified hardware target is one NVIDIA RTX 4090 integrated with GradLab, while the design must remain structurally portable to other CUDA-capable NVIDIA GPUs without slowing the certified path.
- Seed-independent setup artifacts may be excluded from reusable-run timing only when they contain no learned or run-specific state, remain unchanged across cold-start runs, and have their contents and creation costs disclosed.
- Generality and future features must not impose material overhead on certified fast paths.
- `env-Doom-turbo-torch` must pursue world-leading reproducible performance without presenting comparative or superlative claims as achieved unless current workload-equivalent evidence supports them.
- Final certification must use five predeclared cold-start seeds, report every outcome without replacement, and require at least four policies to satisfy transfer acceptance.
- Public comparative evidence must disclose the code, assets, hardware, software, recipes, seeds, timing boundaries, learning outcomes, selected checkpoints, sample counts, statistical analysis, and failures needed for independent reproduction.

### Transfer and parity

- Transfer acceptance must compare each unchanged frozen Doom-Torch-trained policy under stochastic sampling in `env-Doom-turbo-torch` and `env-ViZDoom-turbo`, without fine-tuning or environment-specific policy-facing compensation.
- Transfer acceptance must require statistical equivalence of `player_kills` within a predeclared practical tolerance while treating other behavior signals as regression guardrails.
- Transfer acceptance must not depend on matching a separately ViZDoom-trained policy.
- Deterministic gameplay mechanics must match the reference environment unless an explicitly documented deviation passes low-level parity and transfer acceptance.
- Minor stochastic divergence is acceptable when its distributions remain compatible and it does not materially harm policy transfer.
- Raw fidelity evaluation must precede observation preprocessing and use ViZDoom deathmatch’s 320×240 RGB24 output with the full Doom HUD.
- Pixel-exact rendering is not required, but policy-facing observations must support reliable transfer.
- Documented approximations may remain only when they pass low-level semantic checks and policy-transfer acceptance.
- Domain randomization may improve resilience to bounded rendering and simulation differences but must not conceal material semantic incompatibility.

### Community, compatibility, and content

- Public releases must be open, installable, documented, and accompanied by independently reproducible benchmark recipes and evidence.
- Use `env-Doom-turbo-torch` as the project and GitHub repository name, `env-doom-turbo-torch` as the Python distribution name, and `env_doom_turbo_torch` as the public Python import package; current project content must not use any former project name or import identifier.
- `env-Doom-turbo-torch` must provide the supported deathmatch API exposed by `env-ViZDoom-turbo`.
- Future releases must provide policy-facing semantic parity with ViZDoom across supported environments, maps, actions, observations, signals, rewards, resets, episode semantics, save-state initialization, and multiplayer while retaining Torch-only transition transport and without slowing certified fast paths.
- Its Turbo-compatible reset and step data plane must accept and return Torch tensors; it must not provide a NumPy transition transport.
- The first certified environment is the ViZDoom deathmatch scenario.
- Certified environments must support user-supplied Doom II and Freedoom WADs.
- Future releases must support multiple certified deathmatch WADs without slowing the initial single-scenario fast path.
- The first certification need not cover CPU optimization, multi-GPU scaling, deterministic-policy evaluation, bidirectional transfer, pretrained-policy results, other Doom modes, multiplayer, or multiple certified WAD profiles.

### Multiplayer

- Future releases must support multi-agent training with multiple independently controlled players.
- Future multiplayer must allow matches containing configurable combinations of trainable policies, frozen policies, humans, scripted opponents, and monsters.

### Licensing

- License permissiveness is subordinate to training throughput, semantic parity, and policy transfer.
- `env-Doom-turbo-torch` may reuse compatible reference-engine source when doing so advances the project objective and all provenance and redistribution obligations are satisfied.
