"""Record a standalone env-Doom-turbo-torch policy through the raw 320x240 HUD renderer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
TRAIN_PATH = ROOT / "train.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("standalone_train", TRAIN_PATH)
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:  # pragma: no cover - import invariant
    raise RuntimeError(f"could not load standalone trainer: {TRAIN_PATH}")
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a trained standalone policy using env-Doom-turbo-torch's raw renderer.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--game-seed", type=int, default=3_410_685_839)
    parser.add_argument("--policy-seed", type=int, default=123)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--deterministic", action="store_true")
    return parser


def _encoder(output: Path, fps: float) -> subprocess.Popen[bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "320x240",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-movflags",
        "+faststart",
        "-y",
        str(output),
    )
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main() -> int:
    args = _parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if not args.iwad.is_file():
        raise FileNotFoundError(f"IWAD does not exist: {args.iwad}")
    if not args.scenario.is_file():
        raise FileNotFoundError(f"scenario does not exist: {args.scenario}")
    if args.seconds <= 0.0:
        raise ValueError("seconds must be positive")

    device = torch.device("cuda")
    torch.manual_seed(args.policy_seed)
    torch.cuda.manual_seed_all(args.policy_seed)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if (
        not isinstance(loaded, Mapping)
        or loaded.get("format") != "standalone-env_doom_turbo_torch-ppo-v1"
    ):
        raise ValueError(f"unsupported standalone checkpoint: {args.checkpoint}")

    env_args = argparse.Namespace(
        scenario=args.scenario.resolve(),
        iwad=args.iwad.resolve(),
        num_envs=1,
        wall_contact_damage_scale=1.0,
        compile_engine=False,
    )
    env = train._make_env(env_args, device, num_envs=1)
    fps = 35.0 / train.REFERENCE_RECIPE.frame_skip
    frame_count = round(args.seconds * fps)
    encoder = _encoder(args.output.expanduser().resolve(), fps)
    if encoder.stdin is None:
        env.close()
        raise RuntimeError("ffmpeg did not expose its input pipe")

    policy = train.NatureActorCritic().to(device)
    policy.load_state_dict(loaded["policy_state_dict"])
    policy.eval()
    context_encoder = train.CombatContextEncoder(env.device_info_history_names, device)
    reset_mask = torch.ones(1, dtype=torch.bool, device=device)
    game_seed = torch.tensor([args.game_seed], dtype=torch.int64, device=device)
    observations, _signals = env.reset_device(reset_mask, game_seed)
    context = context_encoder.encode(env.device_info_histories())
    done = False
    kills = 0.0
    decisions = 0
    try:
        for _frame_index in range(frame_count):
            with torch.no_grad():
                if args.deterministic:
                    actions = policy.deterministic_action(observations, context)
                else:
                    actions, _values, _log_probs = policy.act(observations, context)
            transition = env.step_and_reset_device(actions, game_seed + 1)
            frame = env.render_lane(0)
            encoder.stdin.write(frame.tobytes())
            observations = transition.observations
            context = context_encoder.encode(transition.info_histories)
            decisions += 1
            if bool((transition.terminated | transition.truncated)[0].item()):
                kill_index = tuple(env.device_signal_names).index("killcount")
                kills = float(transition.final_signals[0, kill_index].item())
                done = True
                break
    finally:
        encoder.stdin.close()
        env.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the policy playback")

    output = args.output.expanduser().resolve()
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(loaded.get("step", 0)),
        "deterministic_actions": bool(args.deterministic),
        "decisions": decisions,
        "episode_completed": done,
        "fps": fps,
        "game_seed": args.game_seed,
        "kills_if_completed": kills if done else None,
        "output": str(output),
        "policy_seed": args.policy_seed,
        "resolution": [320, 240],
        "seconds": decisions / fps,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
