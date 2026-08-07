# Deathmatch parity and certification

## Reference identity

- Scenario WAD SHA-256: `1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d`
- Scenario CFG SHA-256: `6733112703b3264e5795c5478baea2ed01d3912d5321bda11ac1e3f1377d9d3b`
- Known Doom II IWAD SHA-256: `10d67824b11025ddd9198e8cfc87ca335ee6e2d3e63af4180fa9b8a471893255`

Hashes identify operator-supplied inputs; the files are not distributed here.

## Required deterministic mechanics

Tick/action timing, movement, collision, weapon selection and state, ammo, hitscan/projectile behavior, damage, armor, pickups, monster state machines, kills, player death, episode boundaries, and task signals must match deterministic reference fixtures or have an explicitly accepted deviation.

## Permitted statistical parity

Spawn selection, random damage, monster decisions, and equivalent tie-breaking may use different random streams only when distribution tests and zero-shot policy transfer pass.

## Release gates

1. Differential micro-scenarios pass for all deterministic mechanics.
2. Stochastic outcome distributions stay within declared bounds.
3. At least five GraDOOM training seeds are evaluated unchanged over 100 ViZDoom episodes each.
4. Mean ViZDoom kills is at least 10 and at least 90% of the matched ViZDoom-trained reference.
5. Median wall-clock time to the first passing checkpoint beats the strongest matched baseline by a statistically meaningful margin.
