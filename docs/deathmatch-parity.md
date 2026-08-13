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

## Release gates

1. Differential micro-scenarios pass for all deterministic mechanics.
2. Stochastic outcome distributions stay within declared bounds.
3. At least five GraDOOM training seeds are evaluated unchanged over 100 ViZDoom episodes each.
4. Mean ViZDoom kills is at least 10 and at least 90% of the matched ViZDoom-trained reference.
5. Median wall-clock time to the first passing checkpoint beats the strongest matched baseline by a statistically meaningful margin.
