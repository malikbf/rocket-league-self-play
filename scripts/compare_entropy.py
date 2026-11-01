"""Evaluate a matched entropy-control/treatment pair; no further training."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4


def compare(control: Path, treatment: Path, output: Path) -> None:
    """Compare final checkpoints and shared initial/baseline on ten fixed seeds."""
    import numpy as np
    import torch
    from rlgym.rocket_league.action_parsers import LookupTableAction
    from rocket_league_ai.agents.baselines import DriveToBallPolicy
    from rocket_league_ai.agents.policies import load_policy
    from rocket_league_ai.evaluation.runner import evaluate
    from rocket_league_ai.training.config import load_config

    torch.set_num_threads(1)
    configs = [load_config(run / "config.toml") for run in (control, treatment)]
    left, right = [asdict(config) for config in configs]
    for settings in (left, right):
        settings["experiment"].pop("name")
        settings["training"].pop("entropy_coef")
    if left != right or configs[0].training.entropy_coef != 0.005 or configs[1].training.entropy_coef != 0:
        raise ValueError("Expected matched fresh runs with entropy coefficients 0.005 and 0")
    initial = load_policy(control / "reference_policy.pt")
    other_initial = load_policy(treatment / "reference_policy.pt")
    if not all(torch.equal(value, other_initial.state_dict()[key]) for key, value in initial.state_dict().items()):
        raise ValueError("Initial policy weights differ")
    policies = [("initial", initial)]
    training = {}
    for label, run in (("control", control), ("no_entropy", treatment)):
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        resume = json.loads((run / "resume.json").read_text(encoding="utf-8"))
        if resume["checkpoint"] is not None or result["new_steps"] != 40960:
            raise ValueError("Expected fresh successful 40,960-transition runs")
        checkpoint = run / "checkpoints" / str(result["steps"]) / "PPO_POLICY.pt"
        policies.append((label, load_policy(checkpoint)))
        progress = result["iterations"]
        training[label] = {"run": str(run.resolve()),
                           "metadata": json.loads((run / "metadata.json").read_text(encoding="utf-8")),
                           "touches": sum(row["gameplay"]["ball_touches"] for row in progress),
                           "goals": sum(row["gameplay"]["goals"] for row in progress),
                           "progress": progress}
    policies.append(("drive-to-ball-v1", DriveToBallPolicy(LookupTableAction.make_lookup_table())))
    rows = []
    seeds = tuple(range(1001, 1011))
    for label, policy in policies:
        for start in ("kickoff", "approach"):
            modes = ("argmax",) if label == "drive-to-ball-v1" else ("argmax", "sample")
            for mode in modes:
                report = evaluate(policy, initial, start_mode=start, action_selection=mode, seeds=seeds)
                rows.append({"candidate": label, "start_mode": start, **report})
                print(f"{label:16} {start:8} {mode:6}: touches={report['touches']}, "
                      f"contact={report['touch_episode_rate']:.0%}", flush=True)
    root = Path(__file__).resolve().parents[1]

    def capture(arguments: list[str]) -> str:
        return subprocess.check_output(arguments, cwd=root, text=True, timeout=15).strip()

    runtime = {"commit": capture(["git", "rev-parse", "HEAD"]),
               "git_status": capture(["git", "status", "--short"]),
               "platform": platform.platform(), "python": sys.version,
               "torch": torch.__version__, "cuda_build": torch.version.cuda,
               "numpy": np.__version__, "device": "cpu",
               "requirements": capture([sys.executable, "-m", "pip", "freeze"])}
    output.write_text(json.dumps({"hypothesis": "Entropy bonus limits early policy specialization",
                                  "training_seed": configs[0].experiment.seed, "runtime": runtime,
                                  "training": training, "results": rows}, indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output is not None:
        compare(args.control, args.treatment, args.worker_output)
        return 0
    output = Path(__file__).resolve().parents[1] / "logs" / "experiments" / f"entropy-{uuid4().hex[:8]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Comparison: {output}", flush=True)
    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                str(args.control.resolve()), str(args.treatment.resolve()),
                                "--worker-output", str(output)], start_new_session=os.name != "nt")
    try:
        return process.wait(timeout=300)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False, timeout=10)
        else:
            process.kill()
        process.wait(timeout=10)
        print("Comparison interrupted or exceeded 300 seconds", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
