"""Observation-only driving baseline for calibrating contact evaluation."""
import numpy as np
import torch


class DriveToBallPolicy:
    """Steer toward the ball on the ground; no training, jumping or prediction.

    Uses the pinned DefaultObs 92-vector: ball position 0:3, own position 52:55
    and forward direction 55:58, all in the same team-inverted coordinate frame.
    The caller supplies the installed action table to avoid simulator imports.
    """

    def __init__(self, action_table: np.ndarray):
        if action_table.shape != (90, 8):
            raise ValueError("Expected the 90-by-8 discrete action table")
        self.action_table = action_table.copy()

    def get_output(self, observation: np.ndarray) -> torch.Tensor:
        """Return a deterministic one-hot distribution over existing actions."""
        obs = np.asarray(observation)
        if obs.shape != (92,) or not np.isfinite(obs).all():
            raise ValueError("Expected one finite 92-value DefaultObs vector")
        delta = obs[:2] - obs[52:54]
        forward = obs[55:57]
        angle = float(np.arctan2(forward[0] * delta[1] - forward[1] * delta[0],
                                 np.dot(forward, delta)))
        steer = float(np.sign(angle)) if abs(angle) > 0.1 else 0.0
        boost = float(abs(angle) < 0.15 and np.linalg.norm(delta) * 2300 > 800)
        controls = np.array([1, steer, 0, steer, 0, 0, boost, 0])
        matches = np.flatnonzero(np.all(self.action_table == controls, axis=1))
        if len(matches) != 1:
            raise ValueError("Action table lacks the requested ground control")
        probabilities = torch.zeros(90)
        probabilities[int(matches[0])] = 1
        return probabilities
