# RTX 4090 training optimization

This document records the internal Beast-3 experiment captured on 2026-08-12. It
is a reproducibility note, not a public fastest-training or ViZDoom-parity claim.
The fixed evaluation uses GraDOOM's current environment, so environment parity
and zero-shot policy transfer must be certified separately.

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
GradLab reference policy scored 35.11 mean kills over 100 episodes in ViZDoom
but only 3.59 in the current GraDOOM environment on the corresponding fixed
seed grid. An exact reference-recipe GraDOOM reproduction also peaked at only
7.30 rolling kills by 10.0M steps. These results point to observation or
simulation incompatibility rather than a PPO-throughput limitation.

The retained evidence is under
`/home/tsilva/gradoom-runs/20260813-native-wall025-mature-seed789-goal30` and
`/home/tsilva/gradoom-runs/20260813-wall025-goal30-throughput10x` on Beast-3.
