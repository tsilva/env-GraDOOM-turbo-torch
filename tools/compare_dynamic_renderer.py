"""Rank raw renderer discrepancies along synchronized ViZDoom trajectories.

Unlike ``compare_renderer.py``, this diagnostic advances both environments
through the same action program before sampling frames.  Its defaults exclude
attacks so visual-only hitscan randomness cannot obscure deterministic scene
or movement discrepancies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from compare_behavior import (
    PROGRAMS,
    _action_index,
    _action_matrix,
    _align_give_all,
    _align_pose,
)
from compare_renderer import _match_reference_mugshot

from env_doom_turbo_torch.engine import TorchDeathmatchEngine
from env_doom_turbo_torch.scenario import compile_deathmatch_scenario

NON_FIRING_PROGRAMS = (
    "noop",
    "forward",
    "backward",
    "run-forward",
    "strafe-left",
    "strafe-right",
    "turn-left",
    "turn-right",
)
PROJECTILE_PROGRAM_WEAPONS = {
    "rocket-fire": "SELECT_WEAPON5",
    "plasma-fire": "SELECT_WEAPON6",
}
DYNAMIC_PROGRAMS = PROGRAMS + tuple(PROJECTILE_PROGRAM_WEAPONS)
ITEM_LABEL_CATEGORIES = frozenset(("Ammo", "Armor", "Health", "Powerup", "Weapon"))


def _write_comparison(
    output: Path,
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    difference = torch.abs(reference - actual).mul(3).clamp(0, 255)
    comparison = torch.cat((reference, actual, difference), dim=1).to(torch.uint8)
    subprocess.run(
        (
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "960x240",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-y",
            str(output),
        ),
        input=comparison.cpu().numpy().tobytes(),
        check=True,
    )


def _render_without_weapon(engine: TorchDeathmatchEngine) -> torch.Tensor:
    """Render the raw full-HUD oracle path while ablating only the psprite."""

    (
        frame,
        scene_depth,
        sprite_clip_depth,
        sprite_clip_wall,
        wall_distance,
        blocking_wall,
    ) = engine._render_native_background()
    frame = engine._native_render_hitscan_decals(frame, engine.view_z, scene_depth)
    frame = engine._native_render_sprites(
        frame,
        wall_distance,
        engine.view_z,
        sprite_clip_depth,
        sprite_clip_wall,
        blocking_wall,
    )
    indexed = torch.cat((frame, engine._native_render_hud()), dim=1)
    return engine._native_indexed_to_rgb(indexed)


def _run_case(
    *,
    config: Path,
    iwad: Path,
    engine: TorchDeathmatchEngine,
    seed: int,
    program: str,
    sample_steps: tuple[int, ...],
    frame_skip: int,
    compare_item_occlusion: bool,
    effect_timing_offsets: tuple[int, ...],
    hide_weapon: bool,
    record_object_labels: bool,
    screen_flashes: str,
) -> tuple[list[dict[str, Any]], list[tuple[float, dict[str, Any], torch.Tensor, torch.Tensor]]]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError(
            "compare_dynamic_renderer.py requires the reference vizdoom package"
        ) from exc

    game = vzd.DoomGame()
    config_directory = tempfile.TemporaryDirectory(prefix="env_doom_turbo_torch-vizdoom-renderer-")
    game.load_config(str(config))
    game.set_doom_config_path(str(Path(config_directory.name) / "engine.ini"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_audio_buffer_enabled(False)
    game.set_doom_game_path(str(iwad))
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_weapon(not hide_weapon)
    if screen_flashes != "default":
        game.set_render_screen_flashes(screen_flashes == "on")
    game.set_labels_buffer_enabled(compare_item_occlusion or record_object_labels)
    variables = (
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.POSITION_Z,
        vzd.GameVariable.CAMERA_POSITION_Z,
        vzd.GameVariable.ANGLE,
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.ARMOR,
        vzd.GameVariable.DAMAGECOUNT,
        vzd.GameVariable.HITCOUNT,
        vzd.GameVariable.HITS_TAKEN,
        vzd.GameVariable.DAMAGE_TAKEN,
        vzd.GameVariable.SELECTED_WEAPON,
        vzd.GameVariable.SELECTED_WEAPON_AMMO,
    )
    for variable in variables:
        if variable not in game.get_available_game_variables():
            game.add_available_game_variable(variable)
    game.set_seed(seed)
    game.init()
    try:
        game.new_episode()
        available_buttons = tuple(value.name for value in game.get_available_buttons())
        actions = _action_matrix(available_buttons)
        if program in PROJECTILE_PROGRAM_WEAPONS:
            game.send_game_command("give all")
        engine.reset(
            torch.ones(1, dtype=torch.bool, device=engine.device),
            torch.tensor([seed], device=engine.device),
        )
        initial = {variable.name: float(game.get_game_variable(variable)) for variable in variables}
        _align_pose(engine, initial)

        records: list[dict[str, Any]] = []
        ranked: list[tuple[float, dict[str, Any], torch.Tensor, torch.Tensor]] = []
        sample_set = set(sample_steps)
        last_step = sample_steps[-1]
        for step in range(last_step + 1):
            if step in sample_set:
                state = game.get_state()
                if state is None:
                    raise RuntimeError(
                        f"ViZDoom exposed no state for seed={seed}, program={program}, step={step}"
                    )
                reference = torch.from_numpy(np.asarray(state.screen_buffer).copy()).to(
                    torch.float32
                )
                mugshot = _match_reference_mugshot(engine, reference.to(engine.device))
                actual = (
                    _render_without_weapon(engine)
                    if hide_weapon
                    else engine.render_native_frame(include_hud=True)
                )[0].to(device="cpu", dtype=torch.float32)
                absolute_error = torch.abs(reference - actual)
                flattened = torch.stack((reference.flatten(), actual.flatten()))
                reference_state = {
                    variable.name: float(game.get_game_variable(variable)) for variable in variables
                }
                env_doom_turbo_torch_state = {
                    "ANGLE": float(engine.angle[0]) * 180.0 / np.pi % 360.0,
                    "ARMOR": float(engine.armor[0]),
                    "BONUS_BLEND_COUNT": int(engine.bonus_count[0]),
                    "CAMERA_POSITION_Z": float(engine.view_z[0]),
                    "DAMAGECOUNT": float(engine.player_damagecount[0]),
                    "DAMAGE_BLEND_COUNT": int(engine.damage_count[0]),
                    "DAMAGE_TAKEN": float(engine.player_damage_taken[0]),
                    "HEALTH": float(engine.health[0]),
                    "HITCOUNT": int(engine.player_hitcount[0]),
                    "HITS_TAKEN": int(engine.player_hits_taken[0]),
                    "PLAYER_DEAD": bool(engine.player_dead[0]),
                    "POSITION_X": float(engine.x[0]),
                    "POSITION_Y": float(engine.y[0]),
                    "POSITION_Z": float(engine.z[0]),
                    "SELECTED_WEAPON": int(engine.selected_weapon[0]),
                    "SELECTED_WEAPON_AMMO": float(
                        engine.ammo[0, int(engine.selected_weapon[0]) - 1]
                    ),
                    "VELOCITY_X": float(engine.momentum_x[0]),
                    "VELOCITY_Y": float(engine.momentum_y[0]),
                    "VELOCITY_Z": float(engine.velocity_z[0]),
                }
                record = {
                    "angle": reference_state["ANGLE"],
                    "camera_z": reference_state["CAMERA_POSITION_Z"],
                    "correlation": float(torch.corrcoef(flattened)[0, 1]),
                    "episode_time": int(game.get_episode_time()),
                    "env_doom_turbo_torch_state": env_doom_turbo_torch_state,
                    "mae": float(absolute_error.mean()),
                    "mae_hud": float(absolute_error[208:].mean()),
                    "mae_scene": float(absolute_error[:208].mean()),
                    "matched_mugshot_face_index": mugshot,
                    "program": program,
                    "reference_state": reference_state,
                    "seed": seed,
                    "step": step,
                    "x": reference_state["POSITION_X"],
                    "y": reference_state["POSITION_Y"],
                    "z": reference_state["POSITION_Z"],
                }
                if effect_timing_offsets:
                    saved_projectile_age = engine.projectile_age.clone()
                    saved_impact_tics = engine.projectile_impact_tics.clone()
                    timing_sweep: dict[str, dict[str, float]] = {}
                    try:
                        for state_name in ("flight_age", "impact_remaining"):
                            offset_errors: dict[str, float] = {}
                            for offset in effect_timing_offsets:
                                engine.projectile_age.copy_(saved_projectile_age)
                                engine.projectile_impact_tics.copy_(saved_impact_tics)
                                if state_name == "flight_age":
                                    engine.projectile_age.copy_(
                                        torch.where(
                                            engine.projectile_alive,
                                            torch.clamp_min(saved_projectile_age + offset, 0),
                                            saved_projectile_age,
                                        )
                                    )
                                else:
                                    impact_type = engine.projectile_impact_type.clamp(0, 2)
                                    total_tics = engine.map.projectile_explosion_total_tics[
                                        impact_type
                                    ].to(torch.int32)
                                    engine.projectile_impact_tics.copy_(
                                        torch.where(
                                            saved_impact_tics > 0,
                                            torch.clamp(
                                                saved_impact_tics + offset,
                                                min=1,
                                            ).minimum(total_tics),
                                            saved_impact_tics,
                                        )
                                    )
                                candidate = _render_without_weapon(engine)[0].to(
                                    device="cpu",
                                    dtype=torch.float32,
                                )
                                offset_errors[str(offset)] = float(
                                    torch.abs(
                                        reference[: engine.native_view_height]
                                        - candidate[: engine.native_view_height]
                                    ).mean()
                                )
                            timing_sweep[state_name] = offset_errors
                    finally:
                        engine.projectile_age.copy_(saved_projectile_age)
                        engine.projectile_impact_tics.copy_(saved_impact_tics)
                    record["effect_timing_sweep_mae_scene"] = timing_sweep
                if record_object_labels:
                    record["reference_objects"] = [
                        {
                            "category": label.object_category,
                            "id": int(label.object_id),
                            "name": label.object_name,
                            "position": [
                                float(label.object_position_x),
                                float(label.object_position_y),
                                float(label.object_position_z),
                            ],
                            "velocity": [
                                float(label.object_velocity_x),
                                float(label.object_velocity_y),
                                float(label.object_velocity_z),
                            ],
                        }
                        for label in state.labels
                    ]
                    projectile_alive = engine.projectile_alive[0]
                    projectile_slots = torch.nonzero(projectile_alive).flatten().tolist()
                    record["env_doom_turbo_torch_player_projectiles"] = [
                        {
                            "age": int(engine.projectile_age[0, slot]),
                            "position": [
                                float(engine.projectile_x[0, slot]),
                                float(engine.projectile_y[0, slot]),
                                float(engine.projectile_z[0, slot]),
                            ],
                            "slot": slot,
                            "type": int(engine.projectile_type[0, slot]),
                            "velocity": [
                                float(engine.projectile_velocity_x[0, slot]),
                                float(engine.projectile_velocity_y[0, slot]),
                                float(engine.projectile_velocity_z[0, slot]),
                            ],
                        }
                        for slot in projectile_slots
                    ]
                    impact_slots = (
                        torch.nonzero(engine.projectile_impact_tics[0] > 0).flatten().tolist()
                    )
                    record["env_doom_turbo_torch_player_projectile_impacts"] = [
                        {
                            "position": [
                                float(engine.projectile_x[0, slot]),
                                float(engine.projectile_y[0, slot]),
                                float(engine.projectile_z[0, slot]),
                            ],
                            "remaining_tics": int(engine.projectile_impact_tics[0, slot]),
                            "slot": slot,
                            "type": int(engine.projectile_impact_type[0, slot]),
                        }
                        for slot in impact_slots
                    ]
                if compare_item_occlusion:
                    label_values = [
                        int(label.value)
                        for label in state.labels
                        if label.object_category in ITEM_LABEL_CATEGORIES
                    ]
                    labels = np.asarray(state.labels_buffer)
                    reference_item_mask = torch.from_numpy(
                        np.isin(labels, label_values)
                        if label_values
                        else np.zeros(labels.shape, dtype=np.bool_)
                    )
                    saved_item_available = engine.item_available.clone()
                    engine.item_available.zero_()
                    without_items = engine.render_native_frame(include_hud=True)[0]
                    engine.item_available.copy_(saved_item_available)
                    actual_item_mask = torch.any(
                        actual.to(torch.uint8) != without_items,
                        dim=2,
                    )
                    intersection = actual_item_mask & reference_item_mask
                    union = actual_item_mask | reference_item_mask
                    non_item_mask = ~union
                    scene_non_item_mask = non_item_mask[: engine.native_view_height]
                    intersection_pixels = torch.sum(intersection)
                    union_pixels = torch.sum(union)
                    record.update(
                        {
                            "item_actual_only_pixels": int(
                                torch.sum(actual_item_mask & ~reference_item_mask)
                            ),
                            "item_actual_pixels": int(torch.sum(actual_item_mask)),
                            "item_intersection_over_union": float(
                                torch.where(
                                    union_pixels > 0,
                                    intersection_pixels / union_pixels.clamp_min(1),
                                    torch.ones_like(union_pixels),
                                )
                            ),
                            "item_reference_only_pixels": int(
                                torch.sum(reference_item_mask & ~actual_item_mask)
                            ),
                            "item_reference_pixels": int(torch.sum(reference_item_mask)),
                            "mae_non_items": float(absolute_error[non_item_mask].mean()),
                            "mae_scene_non_items": float(
                                absolute_error[: engine.native_view_height][
                                    scene_non_item_mask
                                ].mean()
                            ),
                        }
                    )
                records.append(record)
                ranked.append(
                    (
                        record.get("mae_non_items", record["mae"]),
                        record,
                        reference,
                        actual,
                    )
                )
            if step == last_step:
                break
            if program in PROJECTILE_PROGRAM_WEAPONS:
                if step == 0:
                    action = actions[0]
                elif step == 1:
                    action = [0.0] * len(available_buttons)
                    action[available_buttons.index(PROJECTILE_PROGRAM_WEAPONS[program])] = 1.0
                else:
                    action = actions[1]
            else:
                action = actions[_action_index(program, step)]
            game.make_action(action, frame_skip)
            engine.step(torch.tensor(action, dtype=torch.bool, device=engine.device))
            if program in PROJECTILE_PROGRAM_WEAPONS and step == 0:
                _align_give_all(engine)
        return records, ranked
    finally:
        game.close()
        config_directory.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 456, 789, 1_337))
    parser.add_argument(
        "--programs",
        choices=DYNAMIC_PROGRAMS,
        nargs="+",
        default=NON_FIRING_PROGRAMS,
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        nargs="+",
        default=(0, 10, 20, 30, 40, 50),
    )
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--allow-stochastic-state-divergence",
        action="store_true",
        help=(
            "sample after the first ACS spawn for diagnosis; post-spawn image errors "
            "then combine renderer differences with permitted RNG-state divergence"
        ),
    )
    parser.add_argument(
        "--compare-item-occlusion",
        action="store_true",
        help="compare isolated env-Doom-turbo-torch item pixels with ViZDoom's label buffer",
    )
    parser.add_argument(
        "--effect-timing-offsets",
        type=int,
        nargs="+",
        default=(),
        help=(
            "sweep signed player-projectile flight-age and impact-remaining-tic "
            "offsets against synchronized ViZDoom scene pixels"
        ),
    )
    parser.add_argument(
        "--hide-weapon",
        action="store_true",
        help="ablate the first-person weapon in both renderers while retaining the full HUD",
    )
    parser.add_argument(
        "--record-object-labels",
        action="store_true",
        help=(
            "record ViZDoom object labels and matched env-Doom-turbo-torch player-projectile "
            "state for simulation/render timing diagnostics"
        ),
    )
    parser.add_argument(
        "--screen-flashes",
        choices=("default", "off", "on"),
        default="default",
        help=(
            "leave ViZDoom's flash mode untouched or explicitly configure both "
            "renderers with screen flashes disabled/enabled"
        ),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sample_steps = tuple(sorted(set(args.sample_steps)))
    if not sample_steps or sample_steps[0] < 0:
        parser.error("sample steps must contain non-negative values")
    if args.frame_skip <= 0:
        parser.error("frame skip must be positive")
    if sample_steps[-1] * args.frame_skip >= 106 and not args.allow_stochastic_state_divergence:
        parser.error(
            "comparison must stop before the first stochastic ACS monster spawn at episode time 106"
        )
    if args.top_k < 0:
        parser.error("top-k must be non-negative")

    config = args.config.expanduser().resolve()
    scenario_path = args.scenario.expanduser().resolve()
    iwad = args.iwad.expanduser().resolve()
    scenario = compile_deathmatch_scenario(scenario_path, iwad)
    engine = TorchDeathmatchEngine(
        scenario,
        1,
        device=torch.device(args.device),
        frame_skip=args.frame_skip,
        debug_checks=False,
        render_screen_flashes=args.screen_flashes == "on",
    )
    records: list[dict[str, Any]] = []
    ranked: list[tuple[float, dict[str, Any], torch.Tensor, torch.Tensor]] = []
    for seed in args.seeds:
        for program in args.programs:
            case_records, case_ranked = _run_case(
                config=config,
                iwad=iwad,
                engine=engine,
                seed=seed,
                program=program,
                sample_steps=sample_steps,
                frame_skip=args.frame_skip,
                compare_item_occlusion=args.compare_item_occlusion,
                effect_timing_offsets=tuple(dict.fromkeys(args.effect_timing_offsets)),
                hide_weapon=args.hide_weapon,
                record_object_labels=args.record_object_labels,
                screen_flashes=args.screen_flashes,
            )
            records.extend(case_records)
            ranked.extend(case_ranked)
            print(
                f"completed seed={seed} program={program} "
                f"worst_mae={max(record['mae'] for record in case_records):.6f}",
                flush=True,
            )

    ranked.sort(key=lambda item: item[0], reverse=True)
    top = ranked[: args.top_k]
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_mae, record, reference, actual) in enumerate(top, start=1):
            filename = (
                f"rank-{rank:02d}-seed-{record['seed']}-{record['program']}-"
                f"step-{record['step']}.png"
            )
            _write_comparison(output_dir / filename, reference, actual)

    errors = np.asarray([record["mae"] for record in records], dtype=np.float64)
    correlations = np.asarray(
        [record["correlation"] for record in records],
        dtype=np.float64,
    )
    result = {
        "frame_skip": args.frame_skip,
        "first_person_weapon_hidden": args.hide_weapon,
        "mean_correlation": float(correlations.mean()),
        "mean_mae": float(errors.mean()),
        "median_correlation": float(np.median(correlations)),
        "median_mae": float(np.median(errors)),
        "programs": args.programs,
        "records": records,
        "sample_steps": sample_steps,
        "schema": "env_doom_turbo_torch.renderer-parity.dynamic-raw-rgb-hud.v3",
        "screen_flashes": args.screen_flashes,
        "stochastic_phase_included": sample_steps[-1] * args.frame_skip >= 106,
        "stochastic_state_alignment": ["mugshot_face_index"],
        "top": [record for _mae, record, _reference, _actual in top],
    }
    if args.compare_item_occlusion:
        non_item_errors = np.asarray(
            [record["mae_non_items"] for record in records],
            dtype=np.float64,
        )
        result.update(
            {
                "mean_mae_non_items": float(non_item_errors.mean()),
                "median_mae_non_items": float(np.median(non_item_errors)),
                "ranking_metric": "mae_non_items",
            }
        )
        result["item_occlusion"] = {
            "max_actual_only_pixels": max(record["item_actual_only_pixels"] for record in records),
            "max_reference_only_pixels": max(
                record["item_reference_only_pixels"] for record in records
            ),
            "mean_intersection_over_union": float(
                np.mean([record["item_intersection_over_union"] for record in records])
            ),
            "total_actual_only_pixels": sum(
                record["item_actual_only_pixels"] for record in records
            ),
            "total_reference_only_pixels": sum(
                record["item_reference_only_pixels"] for record in records
            ),
        }
    serialized = json.dumps(result, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
