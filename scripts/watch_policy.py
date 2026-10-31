"""Record a bounded checkpoint rollout and open it in the isolated macOS viewer."""
import argparse
import json
import os
import signal
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def latest_policy() -> Path:
    candidates = list((ROOT / "logs/ppo").glob("*/checkpoints/*/PPO_POLICY.pt"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found; run train_smoke.py first")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def record(policy_path: Path, destination: Path, seconds: int) -> None:
    """Save at most 900 states at 15 FPS; no training or renderer imports."""
    import torch
    from rocket_league_ai.agents.policies import choose_action, load_policy
    from rocket_league_ai.environments.factory import build_env
    from rlgym.rocket_league.common_values import BOOST_LOCATIONS

    torch.set_num_threads(1)
    policy = load_policy(policy_path)
    reference = policy_path.parent.parent.parent / "reference_policy.pt"
    opponent = load_policy(reference) if reference.exists() else policy
    random.seed(1001)
    manifest = policy_path.parent / "manifest.json"
    settings = json.loads(manifest.read_text())["environment"] if manifest.exists() else {}
    env = build_env(**settings)
    frames = []

    def physics(obj) -> dict:
        return {"position": obj.position.tolist(), "velocity": obj.linear_velocity.tolist(),
                "angular_velocity": obj.angular_velocity.tolist(),
                "rotation": obj.rotation_mtx.transpose().flatten().tolist()}

    try:
        obs = env.reset()
        for _ in range(seconds * 15):
            obs, _, terminated, truncated = env.step({
                "blue-0": choose_action(policy, obs["blue-0"]),
                "orange-0": choose_action(opponent, obs["orange-0"]),
            })
            state = env.state
            frames.append({"ball": physics(state.ball),
                           "pads": [bool(timer == 0) for timer in state.boost_pad_timers],
                           "cars": [{"id": index + 1, "team": car.team_num,
                                     "hitbox": car.hitbox_type, "boost": float(car.boost_amount),
                                     "physics": physics(car.physics)}
                                    for index, car in enumerate(state.cars.values())]})
            if any(terminated.values()) or any(truncated.values()):
                obs = env.reset()
    finally:
        env.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"fps": 15, "checkpoint": str(policy_path),
                                      "boost_locations": [list(p) for p in BOOST_LOCATIONS],
                                      "frames": frames}, allow_nan=False), encoding="utf-8")
    print(f"Recorded {len(frames)} frames: blue = saved policy; orange = frozen initial opponent", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="PPO_POLICY.pt (default: newest checkpoint)")
    parser.add_argument("--seconds", type=int, default=30, help="Recorded simulation seconds, 1–60")
    parser.add_argument("--watch-seconds", type=int, default=120, help="Playback duration, 1–600")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60 or not 1 <= args.watch_seconds <= 600:
        parser.error("seconds must be 1–60; watch-seconds must be 1–600")
    checkpoint = (args.checkpoint or latest_policy()).resolve()
    replay = ROOT / "artifacts/rlviser/preview.json"
    record(checkpoint, replay, args.seconds)
    if args.record_only:
        return 0
    python = ROOT / ".venv-viewer/bin/python"
    if not python.exists():
        parser.error("Viewer environment is missing; see README macOS viewer setup")
    process = subprocess.Popen([str(python), str(ROOT / "scripts/play_viewer.py"),
                                str(replay), "--seconds", str(args.watch_seconds)],
                               start_new_session=True)
    try:
        return process.wait(timeout=args.watch_seconds + 30)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        print("Playback stopped", file=sys.stderr)
        return 1
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)



if __name__ == "__main__":
    raise SystemExit(main())
