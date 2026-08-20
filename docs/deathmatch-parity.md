# Deathmatch parity and certification

## Reference identity

- Scenario WAD SHA-256: `1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d`
- Scenario CFG SHA-256: `6733112703b3264e5795c5478baea2ed01d3912d5321bda11ac1e3f1377d9d3b`
- Known Doom II IWAD SHA-256: `10d67824b11025ddd9198e8cfc87ca335ee6e2d3e63af4180fa9b8a471893255`

Hashes identify operator-supplied inputs; the files are not distributed here.

Unless explicitly labeled player-attributed, historical "kills" below use
ViZDoom-compatible `KILLCOUNT`. In this single-player scenario that counter also
includes countable monsters killed by monster infighting; it is retained for
provider parity but is not a player-only policy-quality signal.

## Required deterministic mechanics

Tick/action timing, movement, collision, weapon selection and state, ammo, hitscan/projectile behavior, damage, armor, pickups, monster state machines, kills, player death, episode boundaries, and task signals must match deterministic reference fixtures or have an explicitly accepted deviation.

## Raw visual reference

- Visual parity is measured before observation preprocessing at ViZDoom `RES_320X240` in `RGB24` format.
- Raw fidelity captures explicitly enable the full Doom HUD, even though the pinned training config disables it.
- Geometry, palette colors, lighting, weapon/HUD composition, directional sprites, and walk, attack, and death animation timing are compared at matched player poses and native tics.

## Policy observation reference

The release gate applies to the frame consumed by the policy, not only to the
raw renderer. The pinned env-ViZDoom-turbo transform masks the bottom 32 rows in
the 320x240 RGB frame, performs rational RGB area pooling to 84x84, rounds each
RGB channel, then computes grayscale with integer coefficients 77/150/29.

`tools/compare_renderer.py` reports raw and policy-facing metrics together. On
the four pinned static seeds 123, 456, 789, and 1337, env-Doom-turbo-torch's native renderer
plus reference preprocessing reaches 0.999998 mean policy-frame correlation
and 0.00266/255 mean absolute error. The legacy direct-84 renderer reaches only
0.528 correlation and 20.34/255 error on the same cases, so it is explicitly
reported as `approximate` and is not parity evidence.

## Deterministic prefix oracle

`tools/compare_behavior.py` aligns env-Doom-turbo-torch to ViZDoom's randomized initial
pose, then compares player state, motion, weapons, ammo, rewards, and episode
timing over scripted actions. It deliberately stops before episode time 106,
where the first permitted stochastic ACS monster spawn occurs.

## Permitted statistical parity

Spawn selection, random damage, monster decisions, and equivalent tie-breaking may use different random streams only when distribution tests and zero-shot policy transfer pass.

## Current unmodified-mechanics evidence

The 2026-08-13 parity milestone uses frame skip 2, Doom skill 1, and
`wall_contact_damage_scale=1.0` throughout. It is evidence of substantial
progress, not release certification:

- Twelve scripted action programs match the aligned ViZDoom state through the
  complete deterministic prefix before the first permitted ACS spawn.
- The early ACS spawn distribution matches across providers.
- `tools/compare_summoned_monsters.py` compares 64 aligned trials for each of
  the six scenario actor classes. Attack onset, damage, death rate, and motion
  are close; for example, Zombieman mean damage is 4.17 in env-Doom-turbo-torch versus 3.17
  in ViZDoom, ShotgunGuy is 12.94 versus 11.22, and Demon is 9.41 versus 10.53.
- `tools/compare_infighting.py` compares 128 aligned Zombieman/ShotgunGuy
  trials. env-Doom-turbo-torch observes a monster kill in 44.53% of trials at mean decision
  26.82; ViZDoom observes 45.31% at mean decision 27.71. This covers targeting,
  hitscan interception, retaliation, and kill credit in the isolated setup.
- The converted reference policy scores 35.11 mean kills over 100 ViZDoom
  episodes and 28.09 over 100 env-Doom-turbo-torch episodes with the fast native renderer.
  This is useful one-way zero-shot transfer, but env-Doom-turbo-torch retains only 80.0% of
  the source mean and therefore does not yet satisfy the release gate.
- A env-Doom-turbo-torch-adapted checkpoint scores 20.96 in env-Doom-turbo-torch and 34.95 zero-shot in
  ViZDoom over 100 episodes. Both directions retain useful behavior, but their
  performance is not yet similar enough to claim parity.

The retained aggregate evidence is under `/home/tsilva/env_doom_turbo_torch-runs` in
`20260813-summoned-monster-parity64-seed10000.json`,
`20260813-infighting-zombie-shotgun128-aligned-seed10000.json`,
`20260813-source-layer16-sprite1-depth-eval100-n100-seed10000`, and
`20260813-reference-frozenenc-sidedsign-seed789-20m`.

## 2026-08-14 bug-first parity milestone

Two production parity defects were isolated and corrected without changing
damage scales, rewards, episode rules, or policy inputs:

- The fast native renderer omitted every dynamic combat effect. It now renders
  mutually exclusive player and enemy projectiles, impacts and explosions,
  teleport fog, and hitscan puffs with their reference additive or translucent
  composition styles. In an exact-weapon policy-observation comparison, mean
  absolute error fell from 4.134 to 2.678/255, action-distribution KL divergence
  fell from 0.225 to 0.163, and action argmax agreement rose from 66.1% to
  71.1%. A raw plasma-fire comparison with the weapon hidden reaches
  0.382/255 mean absolute error and 0.9885 correlation over 16 frames.
- Monster hitscan autoaim used the raw target midpoint instead of the target
  vertical interval clipped through portal openings. The corrected CUDA path
  returns the clipped aim interval and preserves ViZDoom attack/chase target
  state timing. Across 1,024 aligned Zombieman/ChaingunGuy infighting trials,
  env-Doom-turbo-torch versus ViZDoom records 4.650 versus 4.558 mean damage, 1.082 versus
  1.104 mean hits, 61.82% versus 62.01% kill observation, and 32.06 versus
  32.21 mean first-kill decision. Post-kill damage, previously exactly zero in
  env-Doom-turbo-torch, is now 6.250 versus 5.872.

The untouched converted reference policy now scores 25.23 mean kills over 100
fixed-seed env-Doom-turbo-torch episodes, up from 23.36 immediately before these fixes. The
same policy scores 35.11 in ViZDoom, so env-Doom-turbo-torch retains 71.9% of the source
mean. This confirms that the fixes improve real policy transfer, but it remains
below both the 30-kill training target and the 90% release gate and is not
certification. The combined corrections sustain 22,639 median environment
transitions per second at 2,048 environments on the reference RTX 4090
benchmark, within 1.7% of the effects-disabled implementation.

Reproducible evidence is retained in:

- `/home/tsilva/env_doom_turbo_torch-runs/20260814-effect-ablation-u300-n32-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-optimized-correct-effect-styles-exact-weapon-u300-n32-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-raw-plasma-fire-hide-weapon-seed1337/`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-infighting-zombie-chaingun1024-portal-autoaim-target-state-fix-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-summoned-zombieman1024-noop-d44-autoaim-state-fix-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-render-effects-autoaim-state-reference-eval100-seed10000.jsonl`

## 2026-08-14 missile-spawn and no-autofire follow-up

Synchronized raw-RGB/state traces exposed two additional deterministic
mechanics defects. env-Doom-turbo-torch now performs Doom's `P_CheckMissileSpawn` collision
test at the already-advanced half-step spawn position, including the missile
radius when deriving a two-sided portal's floor and ceiling opening. It also
implements the Rocket Launcher's `WEAPON.NOAUTOFIRE` flag: attack starts held,
so a trigger held before the weapon first reaches Ready must be released before
the first shot, while `A_ReFire` may continue an established firing sequence.
Neither correction changes rewards, damage, observations, episode rules, or
the policy action space.

In the seed-789 plasma oracle, ViZDoom's first impact is at
`(581.610916, 513.577087, -32)` and env-Doom-turbo-torch's corrected CUDA impact is within
1.5e-5 map units; the first impact scene is pixel exact. In the Rocket Launcher
oracle, both providers retain 100 rockets and zero player damage while attack
is held before Ready. Screen-flash on/off/default ablations are pixel identical
for the causal plasma trajectory and rule out flash composition as the source
of the old divergence.

The full Doom-II-backed suite passes 327 tests, with only three optional
Freedoom tests skipped. On the reference RTX 4090 workload, the corrected
native-fused fast path reaches **22,961 median environment transitions/s** at
2,048 environments, versus 22,639 before the corrections. The result therefore
shows no fast-path regression.

Fixed seed-10000 stochastic evaluation does not establish a policy-quality
gain. The untouched converted ViZDoom policy scores **23.20 mean kills** over
100 env-Doom-turbo-torch episodes (median 18, standard deviation 17.21), versus its prior
25.23 env-Doom-turbo-torch measurement and existing 35.11 ViZDoom result. The 4.03M-sample
env-Doom-turbo-torch-adapted checkpoint scores **26.75 mean kills** (median 23, standard
deviation 17.51), versus 27.39 before the corrections and 39.38 in ViZDoom.
The changes are retained because the raw causal behavior is reference-correct
and the fixed-grid shifts are small relative to episode variance, but the
greater-than-or-equal-to-30 and similar-transfer gates remain unmet.

Reproducible evidence is retained in:

- `/home/tsilva/env_doom_turbo_torch-runs/20260814-plasma-seed789-spawn-opening-cuda.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-rocket-seed789-noautofire-parity.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-projectile-spawn-opening-cuda-4seeds.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-projectile-spawn-noautofire-throughput-2048.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-projectile-spawn-noautofire-source-eval100-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-projectile-spawn-noautofire-adapted-eval100-seed10000.jsonl`

## 2026-08-14 blocked-chase missile parity follow-up

Parity localization now stops exact comparisons at the first stochastic event.
Post-RNG trajectory, frame, and sampled-action divergence is expected and is
not bug evidence. Deterministic predicates use pre-stochastic oracles;
stochastic mechanics use isolated high-sample outcome distributions.

Direct ViZDoom probes of both `P_CheckSight` and the complete initial `A_Look`
target-acquisition decision match env-Doom-turbo-torch in all 512 captured ChaingunGuy
initial states. A separate zero-tic `A_CPosAttack` counter preserved the
original map `BEHAVIOR` and matched all 512 initial player/monster states
exactly. Before the fix, 2,048 trials showed that hit and damage yield per
attack already matched, but env-Doom-turbo-torch executed 0.321 more attacks per trial, with
a normal 95% interval of 0.122 to 0.520.

The causal defect was Doom's failed `P_NewChaseDir` state. ViZDoom retains the
pre-decremented negative `movecount` when every direction is blocked and only
permits missile selection when `movecount == 0`. env-Doom-turbo-torch instead reset a failed
count to zero and accepted every non-positive count, making stuck monsters
immediately missile-eligible. env-Doom-turbo-torch now preserves the negative count and
uses the exact equality guard. This changes no damage, reward, observation,
episode, or policy parameters.

Across 4,096 unmodified-scenario trials after the correction, env-Doom-turbo-torch versus
ViZDoom records 29.314 versus 29.065 mean ChaingunGuy damage (difference 0.250,
95% interval -0.423 to 0.922), 6.978 versus 6.969 mean hits (difference 0.009,
interval -0.143 to 0.160), and 94.263% versus 93.359% any-damage observation
(difference 0.903 percentage points, interval -0.140 to 1.947). The previously
significant combat gaps are therefore compatible with zero. The complete
Doom-II-backed suite passes 330 tests, with three optional Freedoom tests
skipped.

Reproducible evidence is retained in:

- `/home/tsilva/env_doom_turbo_torch-runs/20260814-chaingunguy512-direct-initial-sight-oracle-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-chaingunguy512-direct-initial-look-oracle-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-chaingun-attack-probe-final-initial-state-validation512.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-chaingunguy2048-attack-count-parity-validated-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-summoned-chaingunguy4096-noop-parity-uncertainty-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-summoned-chaingunguy4096-post-negative-movecount-fix-seed10000.json`

## 2026-08-14 corrected-engine training and transfer milestone

Fresh on-policy adaptation after the blocked-chase correction clears the
project's 30-kill training milestone without gameplay scaling. The promoted
checkpoint was initialized from the converted GradLab policy lineage and
trained for 4,030,464 env-Doom-turbo-torch transitions with 2,048 environments, 16-step
rollouts, 4,096-sample minibatches, two PPO epochs, learning rate `1e-6`,
zero entropy coefficient, `killcount-v1`, projection-only visual adaptation,
the native-fused renderer, frame skip 2, and
`wall_contact_damage_scale=1.0`. Training took 220.893 seconds end to end on
the RTX 4090, reached 20,144 median steady-state transitions/s, and logged the
GradLab-compatible return, rolling-kill, throughput, and PPO diagnostics to
W&B run `tsilva/VizdoomDeathmatch-v1/tvoqahok` with the
`env_provider:env_doom_turbo_torch` tag.

On the balanced seed-10000 stochastic acceptance grid, the checkpoint scores
**30.47 mean kills over 100 env-Doom-turbo-torch episodes** (median 33, standard deviation
17.84), up from 28.99 for the pre-correction-trained incumbent evaluated on
the same corrected engine. The unchanged checkpoint scores **38.73 mean kills
over 100 env-ViZDoom-turbo episodes** (median 47, standard deviation 19.07). The
checkpoint SHA-256 is
`554eba0f53b0351f844faeb9733bae6dbcc3277147a160f215a3292983347ee0`.
This clears the explicit greater-than-or-equal-to-30 milestone and provides
strong env-Doom-turbo-torch-to-ViZDoom zero-shot transfer, but remains below the stricter
31.78 historical reference target in env-Doom-turbo-torch.

Reverse transfer remains useful but asymmetric. The untouched converted
GradLab/ViZDoom checkpoint at step 463,970,304 scores **25.80 mean kills over
100 corrected env-Doom-turbo-torch episodes** (median 18.5, standard deviation 18.35), up
from 23.20 before the blocked-chase correction. Its SHA-256 is
`ecf8927d13c4ebc72f9f16e130c8e500578dfdf349f819c24c1875f29c6a72dd`.
The remaining reverse-transfer and five-seed release gates are therefore not
claimed complete.

Scaling the same 4M-sample recipe to 4,096 environments did not improve the
end-to-end objective. The environment-only median rose from 22,991 to 24,228
transitions/s, but realistic PPO steady-state throughput was 19,984
transitions/s, the new-shape compile cost was 72.5 seconds, total training
time increased to 270.390 seconds, and its fixed-32 score was only 26.406.
That run is retained in W&B as
`tsilva/VizdoomDeathmatch-v1/mmze89m5`; 2,048 environments remains the
evidence-backed fast training shape for this recipe.

The registered GradLab `sample-factory-v0` reward was also tested as a
conservative refinement of the promoted checkpoint. A 2,031,616-transition
projection-only run used 2,048 environments, learning rate `5e-7`, and no
mechanics changes. It took 111.582 seconds and sustained 20,372 steady-state
transitions/s, but scored only **24.25 mean kills over the fixed 32-episode
screen**, versus 31.8125 for its parent on the identical grid. This agrees with
the earlier matched from-scratch result, where `sample-factory-v0` reached 6.77
fixed-100 kills versus 11.01 for `native-v1`. Dense reward refinement is
therefore rejected for the current policy lineage; the run is retained in W&B
as `tsilva/VizdoomDeathmatch-v1/r65k0vn9`.

A half-sample time-to-quality attempt doubled the selected learning rate to
`2e-6` while keeping the same seed, initializer, 2,048-by-16 rollout shape,
projection-only adaptation, and `killcount-v1`. PPO remained numerically
conservative (approximately 3e-6 to 8e-6 KL with zero clipping), and the
2,031,616-transition run completed in 110.723 seconds, but its fixed-32 score
fell to **22.219 mean kills**. The faster schedule is rejected: low PPO KL did
not make the semantic adaptation equivalent. The W&B run is
`tsilva/VizdoomDeathmatch-v1/f0rkqfkn`.

An exact second-seed replication exposed material training-seed sensitivity.
Changing only the training seed from 4,127 to 6,841 completed the same
4,030,464-transition recipe in 210.989 seconds end to end, at 20,190
steady-state transitions/s. Its fixed-32 screen scored 31.219 mean kills, but
the authoritative fixed-100 result fell to **28.16 mean kills** (median 27,
standard deviation 18.18). The short screen was therefore not sufficient to
establish robustness, and this checkpoint is rejected. The run is retained in
W&B as `tsilva/VizdoomDeathmatch-v1/ksjid9xt`. This is an end-to-end policy
acceptance result, not evidence of a simulator defect: stochastic policy and
monster streams make post-random-event trajectory divergence non-causal.

Two attempts to reduce that sensitivity were rejected. On seed 6,841, halving
the learning rate to `5e-7` and doubling the schedule to 8,028,160 transitions
completed in 425.437 seconds at 19,788 steady-state transitions/s, but the
final fixed-32 score was only 29.656. The checkpoint prioritized from its
largest online rolling cohort, at step 7,208,960, scored only 26.281 fixed-32;
this again demonstrates that synchronized rolling cohorts cannot select
checkpoints. W&B run `tsilva/VizdoomDeathmatch-v1/vsesnkm9` retains the full
schedule.

A head-only refinement of the promoted checkpoint froze the complete visual
encoder and used cached features during PPO updates. It raised update
throughput from approximately 238 thousand to 1.7 million samples/s, but
simulation, rendering, and per-step inference still dominated. End-to-end
training improved only 12.0%, to 194.324 seconds and 22,218 steady-state
transitions/s, while its fixed-32 score fell to 29.719. This candidate is also
rejected; W&B run `tsilva/VizdoomDeathmatch-v1/6cft0zyr` records it.

The promoted lineage's earlier step-3,276,800 checkpoint was tested for faster
time-to-quality. Its fixed-32 mean was 30.344, but the complete fixed-100 mean
was only **27.16** (median 24.5, standard deviation 17.46), so the 4,030,464
step endpoint remains the earliest proven passing checkpoint. Together with
the seed-6,841 result, these false-positive screens establish that 32 episodes
may reject gross failures but cannot support a near-threshold promotion claim.

Full-rollout PPO batching did not remove the seed sensitivity. On seed 6,841,
one 32,768-sample batch per epoch and learning rate `8e-6` replaced eight
4,096-sample batches at `1e-6`, preserving two epochs and approximately the
same aggregate update size with eight-times-larger gradient estimates. KL and
clipping remained comparable, training took 220.192 seconds at 20,306
steady-state transitions/s, and the fixed-32 mean was 29.875. The candidate is
rejected and retained as W&B run `tsilva/VizdoomDeathmatch-v1/oyk5kb0v`.

Greedy action selection is also rejected as a policy direction. The promoted
checkpoint scored only 18.969 mean kills over the fixed 32-episode grid with
deterministic actions, versus 31.8125 with the normal GradLab-compatible
stochastic sampling. Sampling entropy supplies useful behavioral diversity;
deterministic evaluation is diagnostic only and is not an acceptance result.

Adding a small `0.001` entropy coefficient did not make the exact second-seed
replication clearly competitive. The seed-6,841 run completed 4,030,464
transitions in 210.698 seconds end to end at 20,219 steady-state
transitions/s, but scored 30.844 mean kills on the fixed-32 screen, below the
promoted checkpoint's 31.8125 on the identical grid. Because prior
near-threshold fixed-32 candidates failed the authoritative fixed-100
evaluation, this candidate is rejected without promoting the short screen or
spending a fixed-100 evaluation. W&B run
`tsilva/VizdoomDeathmatch-v1/00oxipa2` retains the result.

Reproducible evidence is retained in:

- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-final100-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-vizdoom-final100-seed10000-pythonpath.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-reference-gradlab-source-post-negative-movecount-fix-eval100-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n4096-lr1e6-4m-seed6841/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n4096-lr1e6-4m-seed6841/eval-final32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-best-sample-factory-projection-lr5e7-2m-seed7717/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-best-sample-factory-projection-lr5e7-2m-seed7717/eval-final32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr2e6-2m-seed4127/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr2e6-2m-seed4127/eval-final32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed6841/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed6841/eval-final32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed6841/eval-final100-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr5e7-8m-seed6841/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr5e7-8m-seed6841/eval-final32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr5e7-8m-seed6841/eval-step7208960-32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-accepted-negative-movecount-frozen-head-killcount-lr1e6-4m-seed6841/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-accepted-negative-movecount-frozen-head-killcount-lr1e6-4m-seed6841/eval-final32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-step3276800-32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-step3276800-100-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-fullbatch-lr8e6-4m-seed6841/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-fullbatch-lr8e6-4m-seed6841/eval-final32-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-final32-seed10000-deterministic-actions.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-ent001-4m-seed6841/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-ent001-4m-seed6841/eval-final32-seed10000.jsonl`

## 2026-08-14 overlapping-sprite renderer parity correction

A wider matched acceptance cohort exposed a remaining closed-loop transfer
gap without being used to localize its cause. The promoted checkpoint scores
24.49 mean kills (standard deviation 17.33) in one env-Doom-turbo-torch episode on each of
100 seed streams, versus 39.20 (standard deviation 19.36) in env-ViZDoom-turbo on
the same provider/game seed grid. The first 16 env-Doom-turbo-torch records are bit-for-bit
identical to the corresponding records from the normal 16-lane evaluator, so
the lower broad-cohort result is not a batch-width simulation defect. The
100-lane protocol is retained as a matched secondary screen; it does not
replace the registered 16-lane GradLab-compatible acceptance protocol.

The gap was not attributed to downstream trajectory divergence. In 256
independent episodes for each of four fixed open-loop action programs, normal
95% intervals for env-Doom-turbo-torch-minus-ViZDoom incoming damage, hit count, episode
length, and their per-decision rates all include zero. Mean-kill intervals
exclude zero only for `noop` (0.332, interval 0.006 to 0.658) and
`forward-fire` (0.656, interval 0.103 to 1.210), where env-Doom-turbo-torch obtains more
kills. This does not support a general claim that env-Doom-turbo-torch combat is harsher;
it moves localization to the controlled observation-policy loop.

Rendering the exact and native-fused observations from each of 9,600
identical env-Doom-turbo-torch states, then passing both through the same promoted policy,
found that the fused renderer retained only the nearest horizontally
overlapping actor for each screen column. A transparent or vertically short
pickup, corpse, or projectile could therefore erase a live enemy behind it.
This is a deterministic rendering defect: Doom's masked-sprite composition
requires farther opaque texels to remain visible through uncovered foreground
texels.

The CUDA compositor now selects the two nearest candidates once and paints
them far-to-near in registers in one Triton launch. A regression constructs a
transparent foreground sprite and an opaque farther sprite and verifies that
the latter remains visible. On the same 9,600 paired states, frame MAE falls
from 2.7303 to 2.0367, exact-to-fast policy KL from 0.13254 to 0.07933
(-40.1%), argmax agreement rises from 71.83% to 78.04%, and feature cosine
from 0.95075 to 0.96719. The final fused implementation is aggregate-identical
to the independently tested two-pass prototype. Its 2,048-environment median
renderer benchmark is 25,391 environment transitions/s, 10.6% faster than
the former one-layer implementation's 22,961 and 38.1% faster than the
correct but launch-bound two-pass prototype's 18,385.

The unchanged promoted policy improves from 24.49 to **27.60 mean kills** on
the identical broad 100-seed env-Doom-turbo-torch cohort (+3.11, +12.7%) after only the
renderer correction. This is direct end-to-end evidence that the defect was
material, although the result remains below both the 30-kill goal and the
matched ViZDoom mean. A fresh 4,030,464-transition projection-only adaptation
used the otherwise identical accepted recipe and no mechanics scaling. It
completed in 192.963 training seconds at 21,570 steady-state transitions/s,
7.1% faster than the former 20,144, and is retained in W&B as
`tsilva/VizdoomDeathmatch-v1/nf6oyjsi`. Its fixed broad-100 result is only
26.40 mean kills (median 22.5, standard deviation 18.52), so the new
checkpoint is rejected; noisy online rolling values are not used to override
the fixed evaluation.

The registered 16-lane, 100-episode seed-10000 evaluation was also rerun for
the unchanged promoted checkpoint under the corrected compositor. It scores
**25.77 mean kills** (median 22, standard deviation 16.98), down from 30.47
under the defective one-layer renderer, and therefore no longer clears the
30-kill milestone. This does not contradict the improvement on the broad
100-lane cohort: policy sampling and subsequent gameplay RNG diverge after
the first changed observation, and each balanced lane consumes a different
sequence of episode seeds. Both fixed end-to-end results are acceptance
evidence; neither downstream trajectory is causal localization evidence. The
former checkpoint remains useful transfer evidence but is no longer promoted
as a current-renderer passing policy.

A controlled effect ablation narrows the remaining observation discrepancy.
Across the same 9,600 identical states, hiding projectiles, impacts, teleport
fog, hitscan puffs, and persistent hitscan decals in both renderers reduces
exact-to-fast KL from 0.07933 to 0.04967 and frame MAE from 2.0367 to 1.4408.
The combined effect representation is therefore a material next localization
target. A decal-only ablation rules out the conspicuous but low-impact missing
fast decal pass: removing decals changes the reference policy by only 0.00054
KL and 0.0100 frame MAE, while the fast frame is unchanged. Future work should
isolate projectile, impact, fog, and puff rendering rather than spend the hot
path budget on wall chips.

The complete Doom-II-backed suite passes 335 tests after the correction, with
the three optional Freedoom/alternate-IWAD tests skipped. Reproducible
evidence is retained in:

- `/home/tsilva/env_doom_turbo_torch-runs/20260814-outcome-distributions-all256-post-negative-movecount-uncertainty-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-accepted-policy-current-render-death-ablation-u300-n32-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-accepted-policy-current-render-live-enemy-ablation-u300-n32-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-accepted-policy-current-render-fused-two-sprite-layers-u300-n32-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-final100-env100-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-final100-env100-seed10000-two-sprite-layers.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-vizdoom-final100-env100-seed10000-pythonpath.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-fused-two-sprite-layers-projection-killcount-n2048-lr1e6-4m-seed4127/train.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-fused-two-sprite-layers-projection-killcount-n2048-lr1e6-4m-seed4127/eval-final100-env100-seed10000.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-negative-movecount-projection-killcount-n2048-lr1e6-4m-seed4127/eval-final100-seed10000-fused-two-sprite-layers.jsonl`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-accepted-policy-fused-two-sprite-layers-effect-decal-ablation-u300-n32-seed10000.json`
- `/home/tsilva/env_doom_turbo_torch-runs/20260814-accepted-policy-fused-two-sprite-layers-decal-ablation-u300-n32-seed10000.json`

## Release gates

1. Differential micro-scenarios pass for all deterministic mechanics.
2. Stochastic outcome distributions stay within declared bounds.
3. At least five env-Doom-turbo-torch training seeds are evaluated unchanged over 100 ViZDoom episodes each.
4. Mean ViZDoom kills is at least 10 and at least 90% of the matched ViZDoom-trained reference.
5. Median wall-clock time to the first passing checkpoint beats the strongest matched baseline by a statistically meaningful margin.
