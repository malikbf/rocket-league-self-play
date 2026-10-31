"""Run from the repository root after editable installation; never starts training."""
import argparse
import importlib
from importlib.metadata import version
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="Device for a tiny PyTorch calculation, not PPO training")
    args = parser.parse_args()
    failures = []
    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    if sys.version_info[:2] != (3, 11):
        failures.append("Python 3.11 is required.")
    for module, distribution in [
        ("rocket_league_ai", "rocket-league-ai"), ("torch", "torch"),
        ("numpy", "numpy"), ("rlgym.api", "rlgym-api"),
        ("rlgym.rocket_league", "rlgym-rocket-league"),
        ("RocketSim", "rocketsim"), ("rlgym_ppo", "rlgym-ppo"),
    ]:
        try:
            importlib.import_module(module)
            print(f"PASS {distribution}: {version(distribution)}")
        except Exception as exc:
            failures.append(f"{module}: {exc}")
    try:
        import torch
        from rocket_league_ai.training.devices import select_device
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        device = select_device(args.device, for_ppo=False)
        x = torch.ones((2, 2), device=device, requires_grad=True)
        (x @ x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all().item()
        print(f"PASS PyTorch forward/backward: {device}")
        print(f"Conservative PPO device: {select_device()}")
        print("MPS availability does not establish RLGym-PPO compatibility.")
    except Exception as exc:
        failures.append(f"PyTorch device check: {exc}")
    try:
        from rlgym.rocket_league.sim import RocketSimEngine
        from rlgym_ppo import Learner
        engine = RocketSimEngine()
        engine.close()
        print("PASS RocketSimEngine construction and Learner import (no rollout)")
    except Exception as exc:
        failures.append(f"Simulation/training imports: {exc}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print("Setup checks failed." if failures else "Setup checks passed; no PPO training was tested.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
