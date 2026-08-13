"""Play one GraDOOM deathmatch lane with a keyboard."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gradoom import GraDoomVecEnv
from gradoom.actions import DEATHMATCH_ACTIONS

_ACTION_INDEX = {buttons: index for index, buttons in enumerate(DEATHMATCH_ACTIONS)}
_NEXT_WEAPON = _ACTION_INDEX[("SELECT_NEXT_WEAPON",)]
_PREVIOUS_WEAPON = _ACTION_INDEX[("SELECT_PREV_WEAPON",)]

_CONTROLS = """Controls:
  W / S                 move forward / backward
  A / D                 strafe left / right
  Left / Right          turn left / right
  Space / Left Ctrl     fire
  Shift + W             run forward
  Q / E                 previous / next weapon
  R                     restart with a new seed
  Esc                   quit
"""


@dataclass(frozen=True, slots=True)
class ControlState:
    """Held controls used to select one action from the certified profile."""

    attack: bool = False
    forward: bool = False
    backward: bool = False
    strafe_left: bool = False
    strafe_right: bool = False
    turn_left: bool = False
    turn_right: bool = False
    run: bool = False


def _select_action(controls: ControlState, weapon_action: int | None = None) -> int:
    """Resolve held keys into the closest action in the pinned 17-action table."""

    if weapon_action is not None:
        return weapon_action

    forward = controls.forward and not controls.backward
    backward = controls.backward and not controls.forward
    strafe_left = controls.strafe_left and not controls.strafe_right
    strafe_right = controls.strafe_right and not controls.strafe_left
    turn_left = controls.turn_left and not controls.turn_right
    turn_right = controls.turn_right and not controls.turn_left

    if controls.attack:
        if forward:
            return _ACTION_INDEX[("ATTACK", "MOVE_FORWARD")]
        if backward:
            return _ACTION_INDEX[("ATTACK", "MOVE_BACKWARD")]
        if strafe_left:
            return _ACTION_INDEX[("ATTACK", "MOVE_LEFT")]
        if strafe_right:
            return _ACTION_INDEX[("ATTACK", "MOVE_RIGHT")]
        if turn_left:
            return _ACTION_INDEX[("ATTACK", "TURN_LEFT")]
        if turn_right:
            return _ACTION_INDEX[("ATTACK", "TURN_RIGHT")]
        return _ACTION_INDEX[("ATTACK",)]

    if forward:
        buttons = ("SPEED", "MOVE_FORWARD") if controls.run else ("MOVE_FORWARD",)
        return _ACTION_INDEX[buttons]
    if backward:
        return _ACTION_INDEX[("MOVE_BACKWARD",)]
    if strafe_left:
        return _ACTION_INDEX[("MOVE_LEFT",)]
    if strafe_right:
        return _ACTION_INDEX[("MOVE_RIGHT",)]
    if turn_left:
        return _ACTION_INDEX[("TURN_LEFT",)]
    if turn_right:
        return _ACTION_INDEX[("TURN_RIGHT",)]
    return _ACTION_INDEX[()]


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play GraDOOM's deathmatch-p1-v1 environment as a human.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iwad",
        type=Path,
        help="Doom II or Freedoom IWAD (or set GRADOOM_IWAD)",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        help="ViZDoom deathmatch.wad (or set GRADOOM_DEATHMATCH_WAD)",
    )
    parser.add_argument("--device", help="Torch device; defaults to CUDA when available")
    parser.add_argument("--seed", type=int, default=0, help="initial episode seed")
    parser.add_argument("--scale", type=_positive_int, default=3, help="integer window scale")
    parser.add_argument(
        "--fps",
        type=_positive_float,
        help="displayed environment steps per second; defaults to real-time Doom tics",
    )
    parser.add_argument(
        "--compile-engine",
        action="store_true",
        help="compile the engine with torch.compile (CUDA only)",
    )
    parser.add_argument(
        "--allow-unpinned-scenario",
        action="store_true",
        help="allow a non-certified deathmatch scenario WAD",
    )
    return parser


def _pressed_controls(keys: Any, pygame: Any) -> ControlState:
    return ControlState(
        attack=bool(keys[pygame.K_SPACE] or keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]),
        forward=bool(keys[pygame.K_w] or keys[pygame.K_UP]),
        backward=bool(keys[pygame.K_s] or keys[pygame.K_DOWN]),
        strafe_left=bool(keys[pygame.K_a]),
        strafe_right=bool(keys[pygame.K_d]),
        turn_left=bool(keys[pygame.K_LEFT]),
        turn_right=bool(keys[pygame.K_RIGHT]),
        run=bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]),
    )


def _draw_frame(pygame: Any, screen: Any, frame: Any) -> None:
    height, width, channels = frame.shape
    if channels != 3:
        raise RuntimeError(f"expected an RGB frame, got shape {frame.shape}")
    native = pygame.image.frombuffer(frame.tobytes(), (width, height), "RGB")
    scaled = pygame.transform.scale(native, screen.get_size())
    screen.blit(scaled, (0, 0))
    pygame.display.flip()


def _caption(signal_names: tuple[str, ...], signals: torch.Tensor) -> str:
    values = signals[0].detach().to("cpu").tolist()
    by_name = dict(zip(signal_names, values, strict=True))
    return (
        "GraDOOM | "
        f"kills {int(by_name['killcount'])}  "
        f"health {int(by_name['health'])}  "
        f"armor {int(by_name['armor'])}  "
        f"ammo {int(by_name['selected_weapon_ammo'])}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise SystemExit("play.py requires pygame-ce; run `uv sync --group dev`") from exc

    pygame.init()
    env: GraDoomVecEnv | None = None
    try:
        env = GraDoomVecEnv(
            game="VizdoomDeathmatch-v1",
            scenario=args.scenario,
            rom_path=None if args.iwad is None else str(args.iwad),
            num_envs=1,
            device=args.device,
            transport="torch",
            use_restricted_actions=DEATHMATCH_ACTIONS,
            render_mode="rgb_array",
            obs_copy="unsafe_view",
            obs_resize=(84, 84),
            obs_crop=(0, 32, 0, 0),
            obs_crop_mode="mask",
            obs_crop_fill=0,
            obs_grayscale=True,
            obs_resize_algorithm="area",
            obs_layout="chw",
            frame_skip=2,
            frame_stack=4,
            maxpool_last_two=False,
            noop_reset_max=0,
            use_fire_reset=False,
            sticky_action_prob=0.0,
            reward_clip=False,
            compile_engine=args.compile_engine,
            require_pinned_scenario=not args.allow_unpinned_scenario,
        )
        env.reset(seed=args.seed)

        initial_frame = env.render()
        if initial_frame is None:  # pragma: no cover - explicit render mode invariant
            raise RuntimeError("rgb_array rendering did not produce a frame")
        native_height, native_width, _ = initial_frame.shape
        screen = pygame.display.set_mode((native_width * args.scale, native_height * args.scale))
        pygame.display.set_caption("GraDOOM")
        step_fps = args.fps or env.metadata["render_fps"] / env.frame_skip
        frame_period = 1.0 / step_fps
        next_frame_at = time.perf_counter()
        action = torch.zeros(1, dtype=torch.int64, device=env.device)
        episode_seed = args.seed
        pending_reset = False
        running = True

        print(_CONTROLS)
        _draw_frame(pygame, screen, env.render())
        while running:
            weapon_action: int | None = None
            restart = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        restart = True
                    elif event.key == pygame.K_q:
                        weapon_action = _PREVIOUS_WEAPON
                    elif event.key == pygame.K_e:
                        weapon_action = _NEXT_WEAPON
            if not running:
                break

            if pending_reset or restart:
                episode_seed += 1
                env.reset(seed=episode_seed)
                pending_reset = False

            controls = _pressed_controls(pygame.key.get_pressed(), pygame)
            action.fill_(_select_action(controls, weapon_action))
            transition = env.step_device(action)
            pending_reset = bool((transition.terminated | transition.truncated)[0].item())

            _draw_frame(pygame, screen, env.render())
            pygame.display.set_caption(_caption(env.device_signal_names, transition.signals))
            next_frame_at += frame_period
            delay = next_frame_at - time.perf_counter()
            if delay > 0:
                pygame.time.wait(round(delay * 1_000))
            else:
                next_frame_at = time.perf_counter()
    finally:
        if env is not None:
            env.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
