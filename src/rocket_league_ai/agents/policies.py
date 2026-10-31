"""Inference helpers for the initial library policy; no simulator dependencies."""
from pathlib import Path

import numpy as np
import torch
from rlgym_ppo.ppo import DiscreteFF


def load_policy(path: Path) -> DiscreteFF:
    """Load a project baseline policy on CPU using tensor-only deserialization."""
    # Constructing a network must not advance the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        policy = DiscreteFF(92, 90, (32, 32), "cpu")
    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    policy.eval()
    if not all(torch.isfinite(parameter).all().item() for parameter in policy.parameters()):
        raise ValueError("Checkpoint contains nonfinite policy parameters")
    return policy


@torch.no_grad()
def choose_action(policy: DiscreteFF, observation: np.ndarray) -> np.ndarray:
    """Choose the most likely control for one car, without changing the policy."""
    probabilities = policy.get_output(observation)
    if probabilities.shape != (90,) or not torch.isfinite(probabilities).all().item():
        raise ValueError("Expected one finite 90-action probability vector")
    return np.array([int(probabilities.argmax().item())], dtype=np.int64)
