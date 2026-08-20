# RTX 4090 training optimization

This document records the internal Beast-3 experiment captured on 2026-08-12. It
is a reproducibility note, not a public fastest-training or ViZDoom-parity claim.
The fixed evaluation uses GraDOOM's current environment, so environment parity
and zero-shot policy transfer must be certified separately.

Unless a result is explicitly labeled player-attributed, historical "kills" in
this document are ViZDoom-compatible `KILLCOUNT`: in single-player deathmatch it
also counts countable monsters killed by monster infighting. Those values are
not player-only policy-quality measurements.

## Acceptance protocol

- Hardware: one NVIDIA GeForce RTX 4090.
- Assets: Doom II IWAD SHA-256
  `10d67824b11025ddd9198e8cfc87ca335ee6e2d3e63af4180fa9b8a471893255`
  and deathmatch scenario SHA-256
  `1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d`.
- Environment: 17 actions, frame skip 2, four 84x84 grayscale frames, Doom skill 1,
  and the `native-v1` scenario reward.
- Observation conversion: the exact ViZDoom `GRAY8` conversion, byte-truncated
  `0.21 R + 0.72 G + 0.07 B`.
- Evaluation: 100 stochastic episodes, 16 balanced lanes, evaluation seed 123,
  using protocol
  `standalone-gradoom-deathmatch-checkpoint-eval-v3-balanced-seed-grid`.
- Quality gates over from-scratch training seeds 123, 456, and 789: every mean at
  least 10 kills, three-seed median at least 12.59, and best mean at least 15.41.
- Throughput gate: at least 134,184 workload-equivalent steady-state
  transitions/s, ten times the audited 13,418.4 transitions/s baseline.

An earlier experiment used a fixed-point grayscale approximation that was not
ViZDoom-exact. Its checkpoints and the previously reported 12.22/13.26/16.42
kill results are invalid and are not used below.

## Quality-learning recipe

All common phases use FP32 PPO, the Nature CNN, native reward, entropy coefficient
0.01, Torch permutation, the fused optimizer, and compiled policy and engine
paths. Checkpoint resume retains optimizer and RNG state.

| Global steps | Envs x steps | Batch | Epochs | Learning rate | Encoder |
|---|---:|---:|---:|---:|---|
| 0-10,010,624 | 2,048 x 8 | 2,048 | 1 | 1e-3 | trainable |
| 10,010,624-17,006,592 | 2,048 x 8 | 2,048 | 1 | 5e-4 | trainable |
| 17,006,592-24,002,560 | 2,048 x 8 | 2,048 | 1 | 2.5e-4 | trainable |

The common recipe's quality-producing stages sustain 90,483-93,516 steady-state
transitions/s, or 6.74-6.97 times the audited baseline. Their summed measured
training time is 279.2-282.4 seconds for 24.0M transitions; initialization and
the separate fixed evaluation are excluded.

The seed-789 checkpoint selected for the best-policy gate uses an alternate final
phase from the same from-scratch 17,006,592-step state: 2,048 x 16 rollouts,
batch 12,288, two epochs, learning rate 3.333333e-4, and a frozen observation
encoder with cached rollout features. That phase ends at 24,018,944 steps and
sustains 131,908 transitions/s.

The accelerated frozen-encoder path fuses uint8 normalization with the first
convolution in a custom Triton kernel. The environment uses exact bitset sector
classification for this map, disabled imitation avoids imitation-only gathers,
and PPO partial final minibatches allow arbitrary batch shapes without dropping
samples.

## Fixed-evaluation results

The common trainable-encoder schedule produced:

| Training seed | Step | Mean kills | Episode median | Max |
|---:|---:|---:|---:|---:|
| 123 | 24,002,560 | 12.86 | 12 | 40 |
| 456 | 24,002,560 | 12.66 | 10 | 39 |
| 789 | 24,002,560 | 15.25 | 13 | 36 |

The selected seed-789 frozen-encoder branch scores 15.76 mean kills, median 14,
and maximum 43 at step 24,018,944. Using it with the two common-recipe results
gives a minimum seed mean of 12.66, a three-seed median of 12.86, and a best mean
of 15.76. All internal quality gates therefore pass. The alternate seed-789
branch was tested during tuning and has not yet been validated on holdout
training seeds.

## Ten-times throughput validation

Starting from the selected 15.76-kill checkpoint, the mature-policy validation
uses 4,096 x 16 rollouts, one full-rollout 65,536-sample PPO minibatch, one epoch,
a frozen encoder, and learning rate 1e-9. Existing 2,048 lane episode counters are
preserved and the new stable lane seed streams begin at episode zero.

Three independent process repeats sustain 149,559, 149,656, and 149,601
transitions/s. Their median is **149,601 transitions/s**, 11.15 times the audited
baseline, and the slowest repeat also clears the ten-times gate. The unchanged
fixed seed grid scores exactly 15.76 mean kills before and after the first stage
(median 14, maximum 43), demonstrating that the measured fast path preserves the
selected policy's behavior.

This is a mature steady-state throughput validation with a deliberately
negligible learning rate; it still executes rollout collection, inference, GAE,
the complete PPO loss, backward pass, gradient clipping, and an optimizer step.
It should not be interpreted as a 10x reduction in end-to-end time-to-quality.
The measured quality-producing common recipe is approximately 6.8x the baseline,
while the selected frozen quality phase is 9.83x.

The retained JSONL and checkpoint evidence is under
`/home/tsilva/gradoom-opt.LlmBqk/throughput-v1` on Beast-3. Summary metrics use
`steady_state_after_rollouts=2`; compile and initialization time are excluded from
that steady-state statistic and remain present separately in emitted records.
The complete lineage and acceptance audit is
`audit-grayexact-goal-01.json` in that directory.

## Sample Factory reward comparison

A controlled exact-grayscale seed-789 trial changed only the training reward from
`native-v1` to the registered GradLab `sample-factory-v0` reward during the common
0-10,010,624 phase. Evaluation always reports scenario-native kills on the same
fixed 100-episode grid.

| Reward | Step | Mean kills | Median | Max | Steady transitions/s |
|---|---:|---:|---:|---:|---:|
| `native-v1` | 10,010,624 | 11.01 | 9 | 41 | 92,709 |
| `sample-factory-v0` | 10,010,624 | 6.77 | 6 | 27 | 93,519 |

The Sample Factory reward produces 38.5% fewer mean kills in this matched test,
while throughput differs by less than 1%. Native reward therefore remains the
selected optimization reward. This single native-tuned schedule does not rule
out a separately tuned learning rate, reward scale, or longer schedule for
`sample-factory-v0`.

## Experimental greater-than-30-kill curriculum

A 2026-08-13 follow-up reached the internal 30-kill target by using the
experimental wall-contact damage control. This is not a parity or transfer
result: `wall_contact_damage_scale=0.25` changes enemy damage while the player
touches blocking geometry, and the default remains `1.0`.

The selected branch resumed the seed-789 exact-grayscale lineage at step
25,341,952 and used 256 environments x 16 steps, batch size 512, two epochs,
learning rate 3.125e-5, entropy coefficient 0.003, and `native-v1`. It reached a
peak 100-episode rolling mean of **37.02 kills** at step 31,236,096 and ended at
31.09 kills at step 34,000,896. The 8.66M-transition branch took 194.5 seconds
of measured training time and sustained 51,445 transitions/s. The W&B run is
`jppzf0hs` in `tsilva/VizdoomDeathmatch-v1`.

The selected step-32,681,984 checkpoint scored 30.85 mean kills, 31.5 median,
and 62 maximum on the fixed 100-episode seed-123 evaluation, with 1,382.83 mean
episode length. The same checkpoint under default damage scored only 21.06 mean
kills and 995.95 mean episode length. A subsequent 0.5-damage curriculum stage
ended at 21.42 rolling kills and was rejected. The greater-than-30 result must
therefore remain labeled as an experimental mechanics result.

The high-throughput preservation check resumed the selected checkpoint with
4,096 environments x 16 steps, a single 65,536-transition minibatch and epoch,
a frozen visual encoder, the fused uint8 first-convolution kernel, and learning
rate 1e-9. It sustained **147,349 transitions/s**, or **10.98x** the audited
13,418.4-transition/s baseline. Fixed evaluation after this stage reproduced
the selected checkpoint's 30.85 mean, 31.5 median, and 62 maximum kills exactly.
The W&B run is `gt1jwls0`. As with the earlier mature-policy benchmark, this
proves throughput and behavior preservation, not a 10x reduction in
end-to-end time-to-quality.

Parity remains the blocking issue for a certified result. The converted
GradLab reference policy scores 35.11 mean kills over 100 episodes in ViZDoom.
It originally scored only 3.59 with GraDOOM's approximate policy renderer, but
the fast native renderer now recovers 28.09 mean kills, median 26, and maximum
62 without changing gameplay mechanics. This is substantial useful one-way
transfer, but it is only 80.0% of the ViZDoom source mean and remains below the
90% release gate.

The localized observation defect is quantified. At identical static
poses, the raw native RGB renderer averages 0.999998 correlation with ViZDoom,
but the compiled direct-84 policy renderer averages only 0.528 correlation and
20.34/255 MAE across four pinned seeds. Passing the native indexed frame
through the exact ViZDoom-turbo area/grayscale transform restores 0.999998
policy-frame correlation and 0.00266/255 MAE. The retained `native-fused` path
renders native indexed flats, portal walls, actors, and the exact weapon layer
with compact Triton kernels before the reference 84x84 policy transform.

Isolated statistical oracles now cover all six scenario monster classes and a
two-monster infighting setup. Across 128 aligned Zombieman/ShotgunGuy trials,
GraDOOM's kill-observed rate is 44.53% versus 45.31% in ViZDoom, with first-kill
means of 26.82 and 27.71 decisions. These results, together with the matching
pre-spawn deterministic prefix and early ACS spawn distribution, rule out
several broad mechanics hypotheses. Long-horizon stochastic amplification and
remaining observation sensitivity are still plausible.

## Unmodified kill-count continuation milestone

The strongest unmodified GraDOOM checkpoint in this milestone initializes from
the converted reference policy, freezes its visual encoder, and trains the
policy/value head with uniform kill-count reward. It uses 2,048 environments x
16 steps, batch size 4,096, two epochs, learning rate 1e-6, entropy coefficient
zero, frame skip 2, Doom skill 1, and wall damage scale 1.0. At 4,030,464
samples it scores **25.93 mean kills over 100 balanced seed-123 episodes**.

An optimizer- and RNG-state continuation to 8,028,160 total samples added
3,997,696 samples in 182.24 seconds and sustained 22,707 median loop
transitions/s. Its rolling-100 metric peaked at 28.56 at step 7,897,088 and
ended at 23.66. With thousands of synchronously reset lanes, episode completions
arrive in length-sorted cohorts, so that rolling window is retained for GradLab
metric compatibility but is not used as a checkpoint acceptance gate.

The balanced fixed-seed evaluation of the final continuation scores **25.63
mean kills over 100 episodes**, median 22, and maximum 63. It preserves but does
not improve the 25.93 parent, and the required unmodified >=30 result remains
unmet. The continuation checkpoint SHA-256 is
`67989a0ca18c38602cacb5c955bcc68b91c25f433739a6b45fa603e1dcefaae2`.
W&B run `11p5nezd` in `tsilva/VizdoomDeathmatch-v1` carries the mandatory
`env_provider:gradoom` tag and the comparable return, rolling-kill, throughput,
and PPO diagnostics.

The retained evidence is under
`/home/tsilva/gradoom-runs/20260813-native-wall025-mature-seed789-goal30` and
`/home/tsilva/gradoom-runs/20260813-wall025-goal30-throughput10x` for the
experimental result, and under
`/home/tsilva/gradoom-runs/20260813-depth-vizinit-killcount-standardconv-lr1e6-4m`
and
`/home/tsilva/gradoom-runs/20260813-parity-killcount-standardconv-lr1e6-resume8m`
for the unmodified milestone on Beast-3.

## Actor-slide and damage-diagnostic parity follow-up

A 2026-08-14 mechanics follow-up matched Doom's actor-blocked player movement
fallback: when diagonal movement is blocked, the engine now attempts the y-only
move before the x-only move. In a controlled Chaingunner-forward oracle this
raised GraDOOM player displacement from 314.71 to 406.22 map units versus
410.08 in ViZDoom, while mean health damage fell from 23.875 to 15.844 versus
14.641. The fixed 100-episode seed-10000 evaluation of the retained unmodified
checkpoint then scored 25.67 mean kills, 19.5 median, and 63 maximum in
GraDOOM.

The same investigation found a diagnostics-only mismatch around voodoo dolls.
ViZDoom records incoming `HITS_TAKEN` and `DAMAGE_TAKEN` only when the damaged
actor is the real player body. Damage to a health-sharing voodoo doll is not
incoming damage; if the real player caused it, it is instead outgoing
`HITCOUNT` and `DAMAGECOUNT`. GraDOOM now follows those categories while
preserving the existing shared health, armor, thrust, and gameplay effects.
Replaying the same 100 open-loop ViZDoom action traces before and after the
change reproduced kills, returns, episode lengths, and termination rates
exactly, while mean reported incoming damage fell from 112.14 to 92.14 and
mean incoming hits from 27.35 to 24.46. These traces use independent stochastic
monster streams and are a diagnostics check, not a policy-quality gate.

A bounded unmodified refinement initialized from the retained checkpoint and
used 2,048 environments x 16 steps, batch size 4,096, two epochs, learning rate
1e-6, a frozen standard Nature encoder, kill-count reward, frame skip 2, Doom
skill 1, and wall damage scale 1.0. It executed 8,028,160 samples in 450.95
seconds end to end and sustained a median 21,412 steady-state transitions/s.
The GradLab-compatible rolling-100 metric peaked at 43.64 kills at step
4,325,376 but ended at 21.93. A fixed balanced 100-episode seed-10000 gate
scored only **24.69 mean kills** in GraDOOM (median 19.5, maximum 63), versus
**36.40 mean kills** in zero-shot ViZDoom (median 36, maximum 75). The apparent
greater-than-30 training peak was therefore a synchronized-completion cohort,
not a fixed-grid quality breakthrough; this checkpoint is not selected over
the 25.67-mean parent.

The run and comparable PPO diagnostics are in W&B run `jxga8pbb` under
`tsilva/VizdoomDeathmatch-v1` with the `env_provider:gradoom` tag. Its final
checkpoint SHA-256 is
`1fbd6885a516ff2a3afebde5c42e914b65cc7d115b65419893077e66b4b793ba`,
and the retained local evidence is under
`/home/tsilva/gradoom-runs/20260814-actor-slide-refine-killcount-lr1e6-8m-seed1597`.
The unmodified fixed-grid greater-than-30 and bidirectional transfer goals
remain unmet.

The current production-renderer transfer baseline was remeasured after the
death-state rendering and actor-slide parity fixes. On the fixed 100-episode
seed-10000 grid, the untouched converted ViZDoom reference policy scores
**23.36 mean kills** in GraDOOM (median 16.5, maximum 63, mean episode length
1,043.57), versus its existing **35.11 mean kills** in ViZDoom (median 38.5,
maximum 68). Current zero-shot transfer is therefore 66.5% of the source mean.
The earlier 28.09/35.11, or 80.0%, comparison remains useful historical
evidence but is not the current production baseline: its fast observation path
predated restoration of enemy death animations and persistent corpses.

An observation-only follow-up did not justify another renderer change. At
identical trajectory states, the production `native-fused` renderer reaches
82.66% policy-argmax agreement, 0.9707 feature cosine similarity, 1.662/255
frame MAE, and 0.0811 policy KL against the reference renderer. Adding transient
effects and extra sprite layers improved those static similarity measures, but
reduced the untouched policy's full fixed-grid score to 21.35 mean kills and
also reduced renderer throughput. Those experiments were rejected and fully
removed. `tools/benchmark_cuda.py` can now select `approximate`, `native-fused`,
or `reference` observations so future renderer changes can report their cost
under the same benchmark harness.

## Reference-correct missile follow-up

The subsequent missile-spawn and Rocket Launcher no-autofire corrections keep
the workload-equivalent 2,048-environment `native-fused` benchmark at **22,961
median environment transitions/s**, 1.42% above the preceding 22,639 result.
The five measured samples are 24,726, 23,230, 22,270, 22,961, and 22,107
transitions/s after 20 excluded warmup batches.

The old checkpoints do not cross the quality gate under the corrected
mechanics. On the balanced stochastic 100-episode seed-10000 grid, the
untouched source policy scores 23.20 mean kills and the 4.03M-sample adapted
policy scores 26.75. Their existing ViZDoom results are 35.11 and 39.38,
respectively. The next optimization stage must therefore adapt or train on the
corrected environment and pass a fresh fixed-grid gate; a GradLab-compatible
rolling-100 peak alone remains insufficient because synchronized 2,048-lane
completion cohorts are length biased.

## Player-attributed kill optimization

The 2026-08-20 follow-up separates policy quality from monster infighting.
GraDOOM now exposes the GPU-resident `player_killcount` signal, which increments
only when player melee, hitscan, or projectile damage delivers the enemy death.
The ViZDoom-compatible `killcount` signal is unchanged for parity. Standalone
training and evaluation use the new signal for their rolling and headline kill
metrics, while evaluation also emits compatibility `KILLCOUNT` summaries. The
historical 31.78 reference target remains explicitly attached to compatibility
`KILLCOUNT` because the source evidence used that counter.

The fixed comparison uses 100 stochastic episodes over 16 balanced lanes with
seed 10000 and the native-fused renderer. The original retained checkpoint
scores **20.77 player-attributed kills** (median 17.5, maximum 56), versus 23.65
compatibility kills. Thus 2.88 deaths per episode, or 13.9% of the player-only
mean, were hidden by the old metric's infighting credit.

Short 32-episode prefix screens were used only to prune candidates. Direct
player-kill head tuning peaked at 24.03, player kill/hit/damage head tuning at
24.09, and full-batch player-combat refinement at 26.13; none beat its parent on
the identical prefix. The strongest retained full-batch checkpoint scored 29.53
on that screen and advanced to the authoritative gate. Its recipe uses 2,048 x
16 rollouts, one 32,768-sample full-rollout batch, two PPO epochs, learning rate
`8e-6`, zero entropy coefficient, projection-only adaptation, and 4,030,464
transitions.

The selected full-batch checkpoint scores **25.51 player-attributed kills over
100 episodes** (median 26, standard deviation 16.67, maximum 57), a gain of 4.74
kills or **22.8%** over the 20.77 baseline. Its compatibility result is 28.93,
so another 3.42 deaths per episode are correctly excluded as infighting. The
next-best post-missile checkpoint scores 23.47 player kills on the same full
grid. The selected checkpoint SHA-256 is
`f4430d2ff5e08a651c6a52c22e426373c49dd5d55c93b6d6e0caf2906e0baddf`.

The added counter does not introduce a measured throughput regression. A quiet
2,048-environment native-fused benchmark with 20 excluded warmup batches and
five 100-step samples records **23,959 median environment transitions/s**. The
samples are 26,745, 23,142, 23,959, 22,964, and 23,960, 4.35% above the prior
22,961 median under the same protocol.

Player-only evaluation evidence is retained at
`/home/tsilva/gradoom-runs/20260814-negative-movecount-projection-killcount-fullbatch-lr8e6-4m-seed6841/eval-player-kills-final100-native-fused-seed10000.jsonl`;
the baseline and next-best evidence use the same filename in their respective
run directories.
