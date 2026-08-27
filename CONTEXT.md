# GraDOOM

The domain language for comparing GPU-resident GraDOOM environments with `env-ViZDoom-turbo` and measuring repeatable training performance.

## Language

**Parity certification**:
Evidence that every policy in a pinned pretrained policy corpus achieves sufficiently similar outcomes in GraDOOM and `env-ViZDoom-turbo` for one exact WAD profile. A certificate is bound to exact provider revisions, assets, corpus, invariant suite, and evaluation protocol; it is a deliberately bounded proxy for environment parity, not proof of complete semantic parity.
_Avoid_: Training certification, complete parity

**Parity threshold**:
For each corpus policy, the absolute difference between providers' mean `player_killcount` over the same 256 predeclared stochastic episode seeds must not exceed the larger of two kills or ten percent of the `env-ViZDoom-turbo` mean. A 95% bootstrap interval is reported as a diagnostic rather than a gate.
_Avoid_: Statistical equivalence, aggregate corpus threshold

**Parity diagnostics**:
Tools for comparing low-level mechanics, behavior traces, and raw rendering between providers to locate simulation defects that may also impair training performance. Their results guide investigation but are not parity-certification gates.
_Avoid_: Certification gate, complete parity suite

**Player-attributed kill signal**:
The policy-quality signal exposed as `player_killcount` in both GraDOOM and `env-ViZDoom-turbo`, counting kills credited to the evaluated player while excluding kills between enemies.
_Avoid_: `killcount`, `player_kills`

**Compatibility kill signal**:
The ViZDoom-compatible `KILLCOUNT` signal, exposed as `killcount`, whose value may include kills between enemies and therefore does not isolate the evaluated player's performance.
_Avoid_: `player_killcount`, player-attributed kills

**Pretrained policy corpus**:
A versioned set of hashed, frozen stochastic policies selected before certification evaluation, including at least one policy trained in GraDOOM and one trained in `env-ViZDoom-turbo`.
_Avoid_: Training seeds, adapted policies

**WAD profile**:
The exact IWAD and PWAD hashes, map, skill, scenario configuration, action mode, and observation preprocessing used unchanged in both providers for a parity comparison.
_Avoid_: WAD family, scenario name

**Training benchmark**:
A five-seed reproducible cold-start training evaluation measuring reusable time to a mean `player_killcount` of 30 over 100 predeclared held-out GraDOOM episodes on the certified Freedoom2 deathmatch profile. Every outcome is reported without replacement, success requires at least four seeds to reach the threshold, and the summary is median time-to-threshold.
_Avoid_: Parity certification

**GPU-resident training path**:
Simulation, transitions, rollout state, inference, losses, optimizer state, and parameter updates that remain on the GPU; CPU control, logging, bootstrap compilation, and checkpoint I/O are outside the per-step data path.
_Avoid_: GPU-only process

**Reusable training time**:
The wall-clock time that recurs on repeated training runs after permitted one-time bootstrap work has been completed.
_Avoid_: First-run time, bootstrap time

**Bootstrap work**:
One-time, run-independent preparation whose persistent output is reused unchanged across repeated training runs and contains no learned or run-specific state. Recurring initialization, compilation, capture, and warm-up are not bootstrap work.
_Avoid_: Training time, warm start
