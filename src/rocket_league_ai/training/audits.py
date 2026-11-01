"""Read-only checks of collected PPO data before optimization."""
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from rlgym_ppo import Learner


def audit_rollout(learner: "Learner", experience: tuple) -> dict:
    """Record signal scale and verify action/log-prob alignment, without updates."""
    states, actions, old_log_probs, rewards, _, dones, truncated = experience
    policy = learner.ppo_learner.policy
    errors = []
    with torch.no_grad():
        for offset in range(0, len(states), 512):
            obs = torch.as_tensor(states[offset:offset + 512], dtype=torch.float32, device=learner.device)
            acts = torch.as_tensor(actions[offset:offset + 512], device=learner.device).reshape(-1, 1)
            current, _ = policy.get_backprop_data(obs, acts)
            old = np.asarray(old_log_probs[offset:offset + 512]).reshape(-1)
            errors.extend(np.abs(current.cpu().numpy().reshape(-1) - old).tolist())
    error = float(max(errors))
    if not np.isfinite(error) or error > 1e-4:
        raise RuntimeError(f"Collected action log probabilities disagree with policy: {error}")
    buffer = learner.experience_buffer

    def statistics(values) -> dict:
        data = np.asarray(values, dtype=np.float64)
        if not np.isfinite(data).all():
            raise RuntimeError("Nonfinite PPO audit data")
        return {"mean": float(data.mean()), "std": float(data.std()),
                "min": float(data.min()), "max": float(data.max()),
                "nonzero_fraction": float(np.count_nonzero(data) / data.size)}

    return {"transition_count": len(states), "buffer_size": len(buffer.rewards),
            "max_action_log_probability_error": error,
            "rewards": statistics(rewards),
            "advantages": statistics(buffer.advantages.detach().cpu().numpy()),
            "value_targets": statistics(buffer.values.detach().cpu().numpy()),
            "termination_count": int(np.count_nonzero(dones)),
            "truncation_count": int(np.count_nonzero(truncated)),
            "entropy_coef": learner.ppo_learner.ent_coef}
