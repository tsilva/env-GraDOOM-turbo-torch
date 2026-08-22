# ViZDoom parity plan

This plan sequences the approved ViZDoom parity requirement in `SPECS.md`. It is an execution plan, not a second source of stakeholder requirements. GitHub issues track active work; update this plan only when milestone order, gates, or the upstream environment catalog changes.

## Outcome

Reach policy-facing semantic parity with ViZDoom while preserving Torch-only, GPU-resident steady-state execution and the separately optimized deathmatch certification path.

An environment is **supported** after its exact identifier or configuration passes the common parity gate. Support is cumulative. An environment is **certified** only through the separate certification requirements in `SPECS.md`; support does not imply certification.

## Execution rule

- Use one exact environment identifier, or one exact config/map/skill tuple, as each milestone's release unit.
- Keep at most one environment milestone in progress.
- Execute only milestones in the active sequence. Wishlist milestones are ineligible until the stakeholder explicitly promotes them into the active sequence.
- Pull shared runtime work into the first environment that needs it; do not create capability-only milestones between environments.
- Begin the next milestone only after the current environment passes every applicable common gate.
- When an environment introduces a feature whose value is undecided, stop that milestone at its decision gate and ask the stakeholder. Continue only work independent of that decision.
- Treat throughput as a design input during implementation: preserve fast-path specialization, avoid obvious transfers and synchronization, and use focused profiling when it informs a design choice. Run the comprehensive throughput evaluation at the milestone gate rather than after every intermediate change.
- Preserve all previously supported environments and pass the deathmatch performance guard before completing every milestone.

## Runtime design

Keep the public `EnvDoomTurboTorchVecEnv` interface small. Put scenario variability behind these deep modules:

| Module | Interface | Implementation hidden behind the seam |
| --- | --- | --- |
| Scenario catalog | Resolve an environment identifier or config path to an immutable source description | Upstream aliases, map and skill selection, asset lookup, hashes, and supported-status metadata |
| Scenario compiler | Compile a source description and IWAD into a `ScenarioProgram` | Config parsing, UDMF and classic map formats, actor definitions, inventory, rewards, episode rules, ACS, render assets, and feature validation |
| Torch Doom runtime | Reset or step a batch using a `ScenarioProgram` and device tensors | Players, actors, weapons, projectiles, sectors, scripts, rewards, termination, signals, and rendering |
| Fast-path compiler | Select or produce a specialized runtime for a program | Deathmatch specialization, static-shape specialization, kernel fusion, CUDA graphs, and cached seed-independent artifacts |
| Parity harness | Compare one environment across reference and Torch adapters | Trace capture, seed alignment, stochastic distribution checks, raw-frame comparisons, performance gates, and evidence reports |

The existing deathmatch engine remains a valid fast-path adapter until a replacement proves semantic and performance non-regression. Scenario-specific behavior belongs in `ScenarioProgram` data or compiled device logic, not in environment-name conditionals in the public environment.

Favor clear generic implementations while preserving the seams needed to specialize measured hot paths. A milestone may iterate through temporarily slower implementations; its final candidate must meet the performance gate before the environment is marked supported.

## Common parity gate

Every environment milestone must complete these steps:

1. Pin the ViZDoom and `env-ViZDoom-turbo` reference versions, scenario/config hashes, IWAD hashes, map, skill, action mode, preprocessing, and seed set.
2. Record the environment's feature delta relative to already supported environments. Route each undecided feature through the decision protocol below.
3. Capture reference reset and action traces before implementation. Include deterministic micro-scenarios for new mechanics and randomized trials for stochastic behavior.
4. Add failing tests through the scenario compiler, Torch runtime, and public environment interfaces. Tests assert observable behavior rather than implementation state.
5. Implement the smallest generic runtime expansion that covers the environment without changing existing environment semantics.
6. Match environment resolution, buttons and action meanings, observation shape and preprocessing, requested game variables, rewards, reset state, episode timing, termination/truncation, and masked-reset behavior.
7. Compare raw reference frames before policy preprocessing. Use exact checks where stable and predeclared tolerances or distribution tests where exact matching is neither required nor stable.
8. Verify that simulation, rendering, rewards, resets, signals, and transitions stay on the configured CUDA device during steady-state stepping.
9. After semantic acceptance is stable, run the full test suite and the milestone performance gate against the pinned A0 baseline. Record the new environment's throughput, rerun the deathmatch semantic, transfer, throughput, memory, and compile-time guards, and resolve any material certified-fast-path regression before completion.
10. Mark the exact environment supported in the public catalog and publish the reproduction command and parity evidence. Leave related identifiers unsupported until their own milestones pass.

Completion means every applicable step passes with retained evidence; a playable environment or passing smoke test is insufficient.

## Feature decision protocol

When a milestone first requires an undecided feature:

1. Report the exact reference behavior and the environments blocked by it.
2. Estimate implementation, validation, steady-state cost, and fast-path risk.
3. Recommend **include**, **defer**, or **exclude**.
4. Ask the stakeholder for the decision. Record approved product scope in `SPECS.md`; record structural choices in an ADR when they constrain later implementation.
5. Keep the environment unsupported while an included feature lacks parity. An approved exclusion must state which environment or interface no longer belongs to the parity target.

Current decisions and remaining gates:

| Feature | First environment that needs it | Status |
| --- | --- | --- |
| Policy-facing `env-ViZDoom-turbo` semantics rather than the complete low-level `DoomGame` Python interface | Basic | Include |
| Save-state initialization, state catalogs, and live snapshots | Basic | Include after the basic default-state path is red/green |
| MultiBinary and delta-button action modes | Basic-MultiBinary and Deathmatch-MultiBinary | Wishlist; do not implement now |
| Enemy and surface variants | Basic-Plus | Wishlist; do not implement now |
| Audio observation buffers | BasicAudio | Wishlist; do not implement now |
| Text notification buffers | BasicNotifications | Wishlist; do not implement now |
| Depth, labels, automap, object, and sector buffers | First custom config requesting each buffer | Defer and ask separately for each buffer family |
| Recording, replay, spectator, and human-window modes | First request | Exclude from the initial parity target unless a research workflow needs them |
| Multiplayer and asynchronous player modes | MultiDuel | Include late, after all active curated single-player scenarios |
| Arbitrary external configs, WADs, ACS, and actor definitions | RocketBasic and the custom corpus | Include incrementally |
| Heretic, Hexen, Strife, and other non-Doom game families present in the underlying engine | First request | Exclude unless explicitly promoted into project scope |
| Pixel-exact rendering | Every environment | Exclude; retain the policy-facing fidelity requirement in `SPECS.md` |

## Active milestone sequence

### Wave A: registered scenario environments

These milestones establish the generic runtime through the active subset of the finite scenario suite. The order minimizes new behavior per milestone.

| Milestone | Environment | Capability introduced or proven |
| --- | --- | --- |
| A0 | `VizdoomDeathmatch-v1` | Freeze the existing semantic, transfer, throughput, memory, and compilation baseline before generic-runtime changes |
| A1 | `VizdoomBasic-v1` | Scenario catalog/compiler seam, fixed player start, stationary one-hit Cacodemon, scripted reward and exit, episode start time, three-button schema |
| A2 | `VizdoomPredictPosition-v1` | Moving Cacodemon, patrol goals, rocket launcher, splash damage, single-shot inventory logic, delayed miss exit |
| A3 | `VizdoomDefendCenter-v1` | Scripted radial spawns, stationary player, chainsaw marine and demon actors, delayed actor respawn |
| A4 | `VizdoomDefendLine-v1` | Imp projectiles, demon pursuit, repeated spawn/removal, infinite-ammo script, unbounded native horizon semantics |
| A5 | `VizdoomTakeCover-v1` | Weaponless survival, stationary invulnerable shooters, periodic random spawns, damage-factor semantics |
| A6 | `VizdoomHealthGathering-v1` | Damaging sectors, randomized medikit spawning, pickup collision, health and survival reward semantics |
| A7 | `VizdoomHealthGatheringSupreme-v1` | Custom actor definitions, custom medikits and poison, randomized start spots, spawn-collision retry |
| A8 | `VizdoomDeadlyCorridor-v1` | Fixed encounter actors, doubled incoming damage, inventory replacement, position-shaped reward, armor-triggered success |
| A9 | `VizdoomMyWayHome-v1` | Random map-point starts, arbitrary initial facing, maze navigation, goal pickup and normal exit |

### Wave B: bundled config-only environments

Treat each config and pinned asset set as one exact environment milestone even when it has no registered Gymnasium identifier.

| Milestone | Environment | Capability introduced or proven |
| --- | --- | --- |
| B1 | `learning.cfg` | Config-path resolution of Basic-equivalent behavior |
| B2 | `simpler_basic.cfg` | Alternate Basic asset parity without scenario-name special casing |
| B3 | `rocket_basic.cfg` | Custom weapon actor definitions and Basic-style rocket task |
| B4 | `multi_duel.cfg` | Multiplayer decision gate, two-player lifecycle, respawn inventory |
| B5 | `multi.cfg` | Asynchronous multiplayer deathmatch, player-number logic, disconnect and respawn scripts |
| B6 | `cig.cfg` | Asynchronous player mode, rocket-only inventory, broader multiplayer map semantics |
| B7 | `oblige.cfg` | External-map configuration without a pinned scenario WAD |

Add other bundled configs only when they expose distinct policy-facing behavior. A config that is an exact alias still receives its own resolution-and-evidence milestone before being marked supported.

### Wave C: active custom-environment conformance

After Wave B, expand arbitrary config/WAD support through a versioned conformance corpus. Each corpus entry is one exact environment defined by config, WAD/IWAD hashes, map, skill, and enabled features, and therefore remains a single-environment milestone.

Choose each next entry to cover one previously unsupported config key, ACS instruction, actor definition, line/sector special, observation buffer, save-state behavior, or multiplayer rule. An entry is eligible only if it does not require a wishlist feature. Full custom-environment support cannot be declared until the applicable wishlist milestones are promoted and completed.

## Wishlist milestones

Wishlist milestones preserve the full-parity destination but are not implementation work now. They require explicit stakeholder promotion into the active sequence. Promotion must retain the one-environment-at-a-time rule and assign new milestone numbers without renumbering completed work.

### Wishlist W1: deferred registered scenarios

| Capability | Environments |
| --- | --- |
| MultiBinary actions | `VizdoomBasic-MultiBinary-v1`, `VizdoomBasicAudio-MultiBinary-v1`, `VizdoomBasicNotifications-MultiBinary-v1`, `VizdoomDeadlyCorridor-MultiBinary-v1`, `VizdoomDefendCenter-MultiBinary-v1`, `VizdoomDefendLine-MultiBinary-v1`, `VizdoomHealthGathering-MultiBinary-v1`, `VizdoomMyWayHome-MultiBinary-v1`, `VizdoomPredictPosition-MultiBinary-v1`, `VizdoomTakeCover-MultiBinary-v1`, `VizdoomDeathmatch-MultiBinary-v1`, `VizdoomHealthGatheringSupreme-MultiBinary-v1` |
| Enemy and surface variants | `VizdoomBasic-Plus-v1`, `VizdoomDefendLine-Plus-v1` |
| Audio observations | `VizdoomBasicAudio-v1` and its MultiBinary variant above |
| Text notifications | `VizdoomBasicNotifications-v1` and its MultiBinary variant above |

When promoted, start with `VizdoomBasic-MultiBinary-v1`, `VizdoomBasic-Plus-v1`, `VizdoomBasicAudio-v1`, and `VizdoomBasicNotifications-v1` as the respective capability frontiers. Each remaining identifier still receives its own milestone and parity evidence.

### Wishlist W2: registered stock-map environments

The upstream catalog generates 680 identifiers from four game/WAD families, their maps, and five skill levels. All use MultiBinary actions, so the entire stock-map wave is blocked until MultiBinary support is promoted and completed. Do not group their support status: each exact identifier gets its own issue, parity evidence, and support transition.

When promoted, process them in this deterministic order:

1. Validate the default-skill frontier (`S3`) one environment at a time: `Freedoom2` MAP01–MAP32, `Doom2` MAP01–MAP32, `Freedoom1` E1M1–E4M9, then `Doom` E1M1–E4M9.
2. For every S3-supported map in the same family/map order, validate `S1`, then `S2`, then `S4`, then `S5`, one identifier per milestone.
3. Use Freedoom environments in public CI. Run paired Doom/Doom II evidence with operator-supplied IWADs and record hashes without distributing commercial assets.

The first stock-map milestone is `VizdoomFreedoom2MAP01-S3-v0`. It must introduce classic Doom map loading, `USE`, doors, switches, exits, sector specials, the seventh weapon slot, and the generic actor/item coverage actually reached by that map. Later maps expand those mechanisms only as their traces demand.

### Wishlist completion frontier

After the deferred registered scenarios and stock-map environments are supported, continue the custom conformance corpus for any action type or observation behavior that was ineligible during active Wave C. Full custom-environment support is reached only when the accepted configuration grammar and feature manifest have no silent fallback: supported features pass parity and unsupported features fail before launch with a precise diagnostic.

## Issue template for each milestone

Create one GitHub issue with the canonical labels and these sections:

- **Environment:** exact identifier or config/map/skill tuple, reference versions, and asset hashes.
- **Entry evidence:** previously supported environments and the deathmatch performance baseline.
- **Feature delta:** every new mechanic, interface behavior, asset type, and configuration option.
- **Decision gates:** unresolved features with recommendation and blocked acceptance checks.
- **Reference corpus:** seeds, action traces, raw observations, signals, rewards, and terminal outcomes.
- **Implementation seam:** the deep module whose implementation changes; public interface changes require explicit justification.
- **Acceptance:** every applicable common parity gate with commands and evidence locations.
- **Non-regression:** full supported-environment matrix plus deathmatch semantic, transfer, throughput, memory, and compilation results.

Label the issue `needs-info` while a stakeholder decision is outstanding, `ready-for-agent` only when entry evidence and decisions are complete, and `ready-for-human` when implementation is complete but operator assets or hardware validation remain.

## Program completion

The parity program is complete when every environment in the pinned upstream catalog is individually supported, the agreed custom-environment conformance surface passes, every included feature has public reproduction evidence, excluded features fail explicitly, and the deathmatch certified fast path still satisfies its semantic and performance requirements.
