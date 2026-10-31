"""Checkpoint persistence for the fixed smoke architecture and normalization."""
from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from rocket_league_ai.training.config import SmokeConfig

if TYPE_CHECKING:
    from rlgym_ppo import Learner


def save_checkpoint(learner: "Learner", config: SmokeConfig) -> Path:
    """Save library networks/optimizers/counters with a compatibility manifest."""
    steps = learner.agent.cumulative_timesteps
    learner.save(steps)
    path = Path(learner.checkpoints_save_folder) / str(steps)
    manifest = {"format": 1, "observations": 92, "actions": 90, "layers": [32, 32],
                "environment": asdict(config.environment), "standardize_obs": False,
                "standardize_returns": False}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def restore_checkpoint(learner: "Learner", path: Path, config: SmokeConfig) -> None:
    """Restore tensors and counters; episodes and RNG streams restart on resume."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    expected = {"format": 1, "observations": 92, "actions": 90, "layers": [32, 32],
                "environment": asdict(config.environment), "standardize_obs": False,
                "standardize_returns": False}
    if manifest != expected:
        raise ValueError("Checkpoint architecture/environment is incompatible with this run")
    ppo = learner.ppo_learner
    for target, filename in (
        (ppo.policy, "PPO_POLICY.pt"), (ppo.value_net, "PPO_VALUE_NET.pt"),
        (ppo.policy_optimizer, "PPO_POLICY_OPTIMIZER.pt"),
        (ppo.value_optimizer, "PPO_VALUE_NET_OPTIMIZER.pt"),
    ):
        target.load_state_dict(torch.load(path / filename, map_location=learner.device, weights_only=True))
    book = json.loads((path / "BOOK_KEEPING_VARS.json").read_text(encoding="utf-8"))
    learner.agent.cumulative_timesteps = book["cumulative_timesteps"]
    learner.agent.average_reward = book["policy_average_reward"]
    ppo.cumulative_model_updates = book["cumulative_model_updates"]
    learner.epoch = book["epoch"]
    learner.ts_since_last_save = book["ts_since_last_save"]
    learner.return_stats.from_json(book["reward_running_stats"])
    for model in (ppo.policy, ppo.value_net):
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise ValueError("Checkpoint has nonfinite model parameters")


def verify_checkpoint(learner: "Learner", path: Path) -> None:
    """Check exact saved network and optimizer tensors against live state."""
    def compare(left, right):
        if isinstance(left, torch.Tensor):
            if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                raise RuntimeError("Checkpoint tensor mismatch")
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise RuntimeError("Checkpoint keys mismatch")
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            if len(left) != len(right):
                raise RuntimeError("Checkpoint length mismatch")
            for first, second in zip(left, right):
                compare(first, second)
        elif left != right:
            raise RuntimeError("Checkpoint value mismatch")

    ppo = learner.ppo_learner
    for target, filename in (
        (ppo.policy, "PPO_POLICY.pt"), (ppo.value_net, "PPO_VALUE_NET.pt"),
        (ppo.policy_optimizer, "PPO_POLICY_OPTIMIZER.pt"),
        (ppo.value_optimizer, "PPO_VALUE_NET_OPTIMIZER.pt"),
    ):
        compare(target.state_dict(), torch.load(path / filename, map_location="cpu", weights_only=True))
    book = json.loads((path / "BOOK_KEEPING_VARS.json").read_text(encoding="utf-8"))
    for actual, key in (
        (learner.agent.cumulative_timesteps, "cumulative_timesteps"),
        (learner.ppo_learner.cumulative_model_updates, "cumulative_model_updates"),
        (learner.epoch, "epoch"),
    ):
        if actual != book[key]:
            raise RuntimeError(f"Checkpoint counter mismatch: {key}")
