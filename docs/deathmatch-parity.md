# Deathmatch parity and certification

## Reference identity

- Scenario WAD SHA-256: `1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d`
- Scenario CFG SHA-256: `6733112703b3264e5795c5478baea2ed01d3912d5321bda11ac1e3f1377d9d3b`
- Known Doom II IWAD SHA-256: `10d67824b11025ddd9198e8cfc87ca335ee6e2d3e63af4180fa9b8a471893255`

Hashes identify operator-supplied inputs; the files are not distributed here.

## Required deterministic mechanics

Tick/action timing, movement, collision, weapon selection and state, ammo, hitscan/projectile behavior, damage, armor, pickups, monster state machines, kills, player death, episode boundaries, and task signals must match deterministic reference fixtures or have an explicitly accepted deviation.

## Raw visual reference

- Visual parity is measured before observation preprocessing at ViZDoom `RES_320X240` in `RGB24` format.
- Raw fidelity captures explicitly enable the full Doom HUD, even though the pinned training config disables it.
- Geometry, palette colors, lighting, weapon/HUD composition, directional sprites, and walk, attack, and death animation timing are compared at matched player poses and native tics.

## Policy observation reference

The release gate applies to the frame consumed by the policy, not only to the
raw renderer. The pinned ViZDoom-turbo transform masks the bottom 32 rows in
the 320x240 RGB frame, performs rational RGB area pooling to 84x84, rounds each
RGB channel, then computes grayscale with integer coefficients 77/150/29.

`tools/compare_renderer.py` reports raw and policy-facing metrics together. On
the four pinned static seeds 123, 456, 789, and 1337, GraDOOM's native renderer
plus reference preprocessing reaches 0.999998 mean policy-frame correlation
and 0.00266/255 mean absolute error. The legacy direct-84 renderer reaches only
0.528 correlation and 20.34/255 error on the same cases, so it is explicitly
reported as `approximate` and is not parity evidence.

## Deterministic prefix oracle

`tools/compare_behavior.py` aligns GraDOOM to ViZDoom's randomized initial
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
  are close; for example, Zombieman mean damage is 4.17 in GraDOOM versus 3.17
  in ViZDoom, ShotgunGuy is 12.94 versus 11.22, and Demon is 9.41 versus 10.53.
- `tools/compare_infighting.py` compares 128 aligned Zombieman/ShotgunGuy
  trials. GraDOOM observes a monster kill in 44.53% of trials at mean decision
  26.82; ViZDoom observes 45.31% at mean decision 27.71. This covers targeting,
  hitscan interception, retaliation, and kill credit in the isolated setup.
- The converted reference policy scores 35.11 mean kills over 100 ViZDoom
  episodes and 28.09 over 100 GraDOOM episodes with the fast native renderer.
  This is useful one-way zero-shot transfer, but GraDOOM retains only 80.0% of
  the source mean and therefore does not yet satisfy the release gate.
- A GraDOOM-adapted checkpoint scores 20.96 in GraDOOM and 34.95 zero-shot in
  ViZDoom over 100 episodes. Both directions retain useful behavior, but their
  performance is not yet similar enough to claim parity.

The retained aggregate evidence is under `/home/tsilva/gradoom-runs` in
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
  GraDOOM versus ViZDoom records 4.650 versus 4.558 mean damage, 1.082 versus
  1.104 mean hits, 61.82% versus 62.01% kill observation, and 32.06 versus
  32.21 mean first-kill decision. Post-kill damage, previously exactly zero in
  GraDOOM, is now 6.250 versus 5.872.

The untouched converted reference policy now scores 25.23 mean kills over 100
fixed-seed GraDOOM episodes, up from 23.36 immediately before these fixes. The
same policy scores 35.11 in ViZDoom, so GraDOOM retains 71.9% of the source
mean. This confirms that the fixes improve real policy transfer, but it remains
below both the 30-kill training target and the 90% release gate and is not
certification. The combined corrections sustain 22,639 median environment
transitions per second at 2,048 environments on the reference RTX 4090
benchmark, within 1.7% of the effects-disabled implementation.

Reproducible evidence is retained in:

- `/home/tsilva/gradoom-runs/20260814-effect-ablation-u300-n32-seed10000.json`
- `/home/tsilva/gradoom-runs/20260814-optimized-correct-effect-styles-exact-weapon-u300-n32-seed10000.json`
- `/home/tsilva/gradoom-runs/20260814-raw-plasma-fire-hide-weapon-seed1337/`
- `/home/tsilva/gradoom-runs/20260814-infighting-zombie-chaingun1024-portal-autoaim-target-state-fix-seed10000.json`
- `/home/tsilva/gradoom-runs/20260814-summoned-zombieman1024-noop-d44-autoaim-state-fix-seed10000.json`
- `/home/tsilva/gradoom-runs/20260814-render-effects-autoaim-state-reference-eval100-seed10000.jsonl`

## 2026-08-14 missile-spawn and no-autofire follow-up

Synchronized raw-RGB/state traces exposed two additional deterministic
mechanics defects. GraDOOM now performs Doom's `P_CheckMissileSpawn` collision
test at the already-advanced half-step spawn position, including the missile
radius when deriving a two-sided portal's floor and ceiling opening. It also
implements the Rocket Launcher's `WEAPON.NOAUTOFIRE` flag: attack starts held,
so a trigger held before the weapon first reaches Ready must be released before
the first shot, while `A_ReFire` may continue an established firing sequence.
Neither correction changes rewards, damage, observations, episode rules, or
the policy action space.

In the seed-789 plasma oracle, ViZDoom's first impact is at
`(581.610916, 513.577087, -32)` and GraDOOM's corrected CUDA impact is within
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
100 GraDOOM episodes (median 18, standard deviation 17.21), versus its prior
25.23 GraDOOM measurement and existing 35.11 ViZDoom result. The 4.03M-sample
GraDOOM-adapted checkpoint scores **26.75 mean kills** (median 23, standard
deviation 17.51), versus 27.39 before the corrections and 39.38 in ViZDoom.
The changes are retained because the raw causal behavior is reference-correct
and the fixed-grid shifts are small relative to episode variance, but the
greater-than-or-equal-to-30 and similar-transfer gates remain unmet.

Reproducible evidence is retained in:

- `/home/tsilva/gradoom-runs/20260814-plasma-seed789-spawn-opening-cuda.json`
- `/home/tsilva/gradoom-runs/20260814-rocket-seed789-noautofire-parity.json`
- `/home/tsilva/gradoom-runs/20260814-projectile-spawn-opening-cuda-4seeds.json`
- `/home/tsilva/gradoom-runs/20260814-projectile-spawn-noautofire-throughput-2048.json`
- `/home/tsilva/gradoom-runs/20260814-projectile-spawn-noautofire-source-eval100-seed10000.jsonl`
- `/home/tsilva/gradoom-runs/20260814-projectile-spawn-noautofire-adapted-eval100-seed10000.jsonl`

## Release gates

1. Differential micro-scenarios pass for all deterministic mechanics.
2. Stochastic outcome distributions stay within declared bounds.
3. At least five GraDOOM training seeds are evaluated unchanged over 100 ViZDoom episodes each.
4. Mean ViZDoom kills is at least 10 and at least 90% of the matched ViZDoom-trained reference.
5. Median wall-clock time to the first passing checkpoint beats the strongest matched baseline by a statistically meaningful margin.
