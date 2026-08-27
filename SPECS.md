## PROJECT PURPOSE

`env-GraDOOM-turbo-torch` is a Torch-native GPU implementation of policy-facing `env-ViZDoom-turbo` behavior and an integrated training system, initially for the deathmatch environment. Its primary outcome is fast, repeatable training on one GPU by maximizing GPU parallelism, keeping the full training path GPU-resident, and minimizing GPU-CPU transfers; over time it should cover most policy-facing behavior supported by `env-ViZDoom-turbo` and remain independently reproducible and useful to expert practitioners.

## PROJECT REQUIREMENTS

### Training performance

- The primary training benchmark must measure reusable wall-clock time to reach a mean `player_killcount` of at least 30 over 100 predeclared held-out GraDOOM episodes on the parity-certified Freedoom2 deathmatch profile.
- Each training benchmark must use five predeclared cold-start seeds with freshly initialized policy and optimizer state, report every outcome without replacement, and require at least four seeds to reach the quality threshold.
- Reusable-run timing must include every recurring action performed by the documented training command, including initialization, per-process or uncached compilation, warm-up, checkpoint evaluation, and checkpoint writing, and must stop at the first passing checkpoint or the predeclared failure budget.
- Only run-independent bootstrap artifacts that persist and are reused unchanged across repeated runs may be excluded from reusable-run timing; their contents, hashes, creation costs, and reuse conditions must be disclosed.
- The benchmark summary must be median time-to-threshold when at least four seeds succeed, while retaining every per-seed time and failure.
- Fixed-time final policy quality and simulated tics or transitions per second must be reported as diagnostic performance measures.
- Simulation, observations, actions, rewards, resets, rollout state, policy inference, losses, optimizer state, and parameter updates must remain GPU-resident during steady-state training.
- CPU work may launch and schedule GPU operations, load configuration, log summaries, create bootstrap artifacts, and write checkpoints, but transitions must not pass through CPU memory during steady-state training.
- The initial training-benchmark hardware target is one NVIDIA RTX 4090 integrated with GradLab, while the design must remain structurally portable to other CUDA-capable NVIDIA GPUs without slowing the target path.
- Generality and future features must not impose material overhead on measured deathmatch fast paths.
- Public performance claims must disclose the code, assets, hardware, software, recipes, seeds, timing boundaries, learning outcomes, checkpoint-selection protocol, sample counts, statistical summaries, and failures needed for independent reproduction.

### Parity certification

- Parity certification is a bounded policy-level proxy for environment parity and must not be presented as proof of complete semantic parity.
- Every certificate must identify the exact GraDOOM and `env-ViZDoom-turbo` revisions, WAD profile, pretrained policy corpus, invariant-suite version, and evaluation protocol; a material change to any identified component requires a new certificate.
- The first parity certificate must cover deathmatch using a pinned Freedoom2 IWAD and deathmatch PWAD.
- Both providers must use byte-identical IWAD and PWAD files and the same map, skill, scenario configuration, action mode, and observation preprocessing during a parity comparison.
- The pretrained policy corpus must be selected and hashed before candidate evaluation and contain at least one frozen stochastic policy trained in GraDOOM and one trained in `env-ViZDoom-turbo`.
- Every corpus policy must run unchanged in both providers without fine-tuning, adaptation, or environment-specific policy-facing compensation.
- Both providers must expose the player-attributed kill signal as `player_killcount`, counting kills credited to the evaluated player while excluding kills between enemies; the ViZDoom-compatible `killcount` signal must not be used as the policy-quality outcome.
- Each corpus policy must be evaluated over the same 256 predeclared stochastic episode seeds in each provider.
- Each corpus policy passes only when the absolute difference between providers’ mean `player_killcount` is no greater than the larger of two kills or ten percent of the `env-ViZDoom-turbo` mean.
- Certification evidence must report a 95% bootstrap interval for each policy comparison as a diagnostic, but the interval is not an additional acceptance gate.
- Certification must include a fast invariant suite covering API shape, device placement, reset and step behavior, episode termination, action meanings, and the player-attributed kill signal.
- The repository must retain reproducible diagnostic tools for comparing low-level mechanics, behavior traces, and raw rendering so agents can locate simulation defects and related training-performance problems; these diagnostics are not certification gates.
- Deathmatch must accept user-supplied Doom II and Freedoom WADs, but every distinct WAD profile requires separate parity evidence and certification.
- Pixel-exact rendering is not required, but policy-facing observations must satisfy the parity-certification outcome.
- Public development releases may remain uncertified when clearly labeled; a parity-certification claim requires current reproducible evidence.

### Compatibility and scope

- Use `env-GraDOOM-turbo-torch` as the project and GitHub repository name, `env-gradoom-turbo-torch` as the Python distribution name, and `gradoom` as the public Python import package; current project content must not use any former project name or import identifier.
- Deathmatch is the only currently committed environment.
- Deathmatch must conform to the policy-facing Turbo vector API of an exactly pinned `env-ViZDoom-turbo` revision, including construction, actions, Torch observations, signals, rewards, reset and step behavior, masked resets, and episode semantics.
- Compatibility does not include the complete low-level ViZDoom `DoomGame` Python interface.
- The reset and step data plane must accept and return Torch tensors and must not provide NumPy transition transport.
- Other environments, multiplayer, and additional policy-facing features are not committed requirements until explicitly approved by the stakeholder.
- Public releases must be open, installable, documented, and clearly state their parity-certification and training-benchmark status.

### Licensing

- License permissiveness is subordinate to training throughput and policy-facing compatibility.
- `env-GraDOOM-turbo-torch` may reuse compatible reference-engine source when doing so advances the project purpose and all provenance and redistribution obligations are satisfied.
