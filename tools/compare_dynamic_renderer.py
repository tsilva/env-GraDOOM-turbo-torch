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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from compare_behavior import PROGRAMS, _action_index, _action_matrix, _align_pose
from compare_renderer import _match_reference_mugshot

from gradoom.engine import TorchDeathmatchEngine
from gradoom.scenario import compile_deathmatch_scenario

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
) -> tuple[list[dict[str, Any]], list[tuple[float, dict[str, Any], torch.Tensor, torch.Tensor]]]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError(
            "compare_dynamic_renderer.py requires the reference vizdoom package"
        ) from exc

    game = vzd.DoomGame()
    game.load_config(str(config))
    game.set_doom_game_path(str(iwad))
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_labels_buffer_enabled(compare_item_occlusion)
    variables = (
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.POSITION_Z,
        vzd.GameVariable.CAMERA_POSITION_Z,
        vzd.GameVariable.ANGLE,
    )
    for variable in variables:
        if variable not in game.get_available_game_variables():
            game.add_available_game_variable(variable)
    game.set_seed(seed)
    game.init()
    try:
        game.new_episode()
        actions = _action_matrix(tuple(value.name for value in game.get_available_buttons()))
        engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([seed]))
        initial = {
            variable.name: float(game.get_game_variable(variable))
            for variable in variables
        }
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
                reference = torch.from_numpy(
                    np.asarray(state.screen_buffer).copy()
                ).to(torch.float32)
                mugshot = _match_reference_mugshot(engine, reference)
                actual = engine.render_native_frame(include_hud=True)[0].to(torch.float32)
                absolute_error = torch.abs(reference - actual)
                flattened = torch.stack((reference.flatten(), actual.flatten()))
                record = {
                    "angle": float(game.get_game_variable(variables[4])),
                    "camera_z": float(game.get_game_variable(variables[3])),
                    "correlation": float(torch.corrcoef(flattened)[0, 1]),
                    "episode_time": int(game.get_episode_time()),
                    "mae": float(absolute_error.mean()),
                    "mae_hud": float(absolute_error[208:].mean()),
                    "mae_scene": float(absolute_error[:208].mean()),
                    "matched_mugshot_face_index": mugshot,
                    "program": program,
                    "seed": seed,
                    "step": step,
                    "x": float(game.get_game_variable(variables[0])),
                    "y": float(game.get_game_variable(variables[1])),
                    "z": float(game.get_game_variable(variables[2])),
                }
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
                            "mae_non_items": float(
                                absolute_error[non_item_mask].mean()
                            ),
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
            action_index = _action_index(program, step)
            game.make_action(actions[action_index], frame_skip)
            engine.step(torch.tensor(actions[action_index], dtype=torch.bool))
        return records, ranked
    finally:
        game.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 456, 789, 1_337))
    parser.add_argument(
        "--programs",
        choices=PROGRAMS,
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
        help="compare isolated GraDOOM item pixels with ViZDoom's label buffer",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    sample_steps = tuple(sorted(set(args.sample_steps)))
    if not sample_steps or sample_steps[0] < 0:
        parser.error("sample steps must contain non-negative values")
    if args.frame_skip <= 0:
        parser.error("frame skip must be positive")
    if sample_steps[-1] * args.frame_skip >= 106 and not args.allow_stochastic_state_divergence:
        parser.error(
            "comparison must stop before the first stochastic ACS monster spawn "
            "at episode time 106"
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
        device=torch.device("cpu"),
        frame_skip=args.frame_skip,
        debug_checks=False,
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
        "mean_correlation": float(correlations.mean()),
        "mean_mae": float(errors.mean()),
        "median_correlation": float(np.median(correlations)),
        "median_mae": float(np.median(errors)),
        "programs": args.programs,
        "records": records,
        "sample_steps": sample_steps,
        "schema": "gradoom.renderer-parity.dynamic-raw-rgb-hud.v1",
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
            "max_actual_only_pixels": max(
                record["item_actual_only_pixels"] for record in records
            ),
            "max_reference_only_pixels": max(
                record["item_reference_only_pixels"] for record in records
            ),
            "mean_intersection_over_union": float(
                np.mean(
                    [record["item_intersection_over_union"] for record in records]
                )
            ),
            "total_actual_only_pixels": sum(
                record["item_actual_only_pixels"] for record in records
            ),
            "total_reference_only_pixels": sum(
                record["item_reference_only_pixels"] for record in records
            ),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
