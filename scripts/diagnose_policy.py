"""Compare a run's initial policy and retained checkpoints without training."""
import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

from rocket_league_ai.agents.policies import load_policy
from rocket_league_ai.evaluation.runner import evaluate


def diagnose(run: Path, output: Path, seed_count: int = 10, baseline_only: bool = False) -> None:
    """Evaluate up to five checkpoints, the reference and driving baseline on CPU."""
    import torch
    from rlgym.rocket_league.action_parsers import LookupTableAction
    from rocket_league_ai.agents.baselines import DriveToBallPolicy

    if type(seed_count) is not int or not 1 <= seed_count <= 20:
        raise ValueError("seed_count must be an integer between 1 and 20")
    seeds = tuple(range(1001, 1001 + seed_count))
    control_names = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
    actions = LookupTableAction.make_lookup_table()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]

    def capture(arguments: list[str]) -> str:
        return subprocess.check_output(arguments, cwd=root, text=True, timeout=15).strip()

    runtime = {"commit": capture(["git", "rev-parse", "HEAD"]),
               "git_status": capture(["git", "status", "--short"]),
               "platform": platform.platform(), "python": sys.version,
               "torch": torch.__version__, "cuda_build": torch.version.cuda,
               "inference_device": "cpu",
               "requirements": capture([sys.executable, "-m", "pip", "freeze"])}
    opponent = load_policy(run / "reference_policy.pt")
    checkpoints = sorted(
        (path for path in (run / "checkpoints").iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    )[-5:]
    if not checkpoints:
        raise ValueError("Run has no saved checkpoints")
    results = []
    candidates = [("initial", run / "reference_policy.pt")] + [
        (path.name, path / "PPO_POLICY.pt") for path in checkpoints
    ]
    candidates = [] if baseline_only else candidates
    candidates.append(("drive-to-ball-v1", None))
    for label, path in candidates:
        policy = load_policy(path) if path is not None else DriveToBallPolicy(actions)
        for start in ("kickoff", "approach"):
            for mode in (("argmax", "sample") if path is not None else ("argmax",)):
                report = evaluate(policy, opponent, start_mode=start, action_selection=mode, seeds=seeds)
                counts = [sum(row["action_counts"][i] for row in report["episodes"]) for i in range(90)]
                report["most_used_action"] = max(range(90), key=counts.__getitem__)
                report["most_used_action_fraction"] = max(counts) / sum(counts)
                report["most_used_action_controls"] = dict(zip(
                    control_names, actions[report["most_used_action"]].tolist()))
                report["mean_action_entropy_nats"] = sum(
                    row["mean_action_entropy_nats"] * row["steps"] for row in report["episodes"]
                ) / sum(row["steps"] for row in report["episodes"])
                results.append({"checkpoint": label, "start_mode": start, **report})
                print(f"{label:>7} {start:8} {mode:6}: touches={report['touches']}, "
                      f"contact={report['touch_episode_rate']:.0%}, "
                      f"distance={report['mean_closest_ball_distance']:.1f}, "
                      f"dominant={report['most_used_action_fraction']:.0%}", flush=True)
    payload = {"format": 2, "run": str(run.resolve()), "runtime": runtime,
               "seeds": list(seeds), "max_steps_per_episode": 150,
               "action_table": [dict(zip(control_names, row.tolist())) for row in actions],
               "distance_units": "RocketSim center-to-center, sampled at reset and decision boundaries",
               "opponent": "frozen initial policy, argmax in both modes",
               "source_metadata": json.loads((run / "metadata.json").read_text(encoding="utf-8")),
               "results": results}
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only calibrate the deterministic drive-to-ball baseline")
    parser.add_argument("--seed-count", type=int, choices=range(1, 21), default=10,
                        metavar="1..20", help="Consecutive seeds from 1001 (default: 10)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    prefix = "calibration" if args.baseline_only else "diagnostics"
    output = args.run / f"{prefix}-{args.seed_count}-seeds.json"
    if args.worker:
        diagnose(args.run, output, args.seed_count, args.baseline_only)
        return 0
    # Hard wall-clock bound, including native simulator failures. No simulator workers.
    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                str(args.run.resolve()), "--seed-count", str(args.seed_count), "--worker"]
                               + (["--baseline-only"] if args.baseline_only else []),
                               start_new_session=os.name != "nt")
    try:
        return process.wait(timeout=300)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=10, check=False)
        else:
            process.kill()
        process.wait(timeout=10)
        print("Diagnostics interrupted or exceeded 300 seconds", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
