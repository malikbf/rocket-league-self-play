"""Rollout alignment checks must reject corrupt data without modifying policy."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from rlgym_ppo.ppo import DiscreteFF

from rocket_league_ai.training.audits import audit_rollout


class AuditTests(unittest.TestCase):
    def test_alignment_signal_statistics_and_no_side_effects(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            policy = DiscreteFF(92, 90, (32, 32), "cpu")
        states = np.zeros((4, 92), dtype=np.float32)
        actions = np.array([[0], [1], [2], [3]])
        with torch.no_grad():
            log_probs, _ = policy.get_backprop_data(states, torch.as_tensor(actions))
        buffer = SimpleNamespace(rewards=torch.zeros(4), advantages=torch.tensor([-1., 0., 1., 2.]),
                                 values=torch.tensor([0., 1., 2., 3.]))
        learner = SimpleNamespace(device="cpu", experience_buffer=buffer,
                                  ppo_learner=SimpleNamespace(policy=policy, ent_coef=0.005))
        experience = (states, actions, log_probs.numpy().reshape(-1), np.array([0., 0., 1., 0.]),
                      states, np.array([0, 0, 0, 1]), np.zeros(4))
        before = {key: value.clone() for key, value in policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = audit_rollout(learner, experience)
        self.assertEqual(report["max_action_log_probability_error"], 0)
        self.assertEqual(report["rewards"]["nonzero_fraction"], 0.25)
        self.assertEqual(report["advantages"]["mean"], 0.5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(before[key], value) for key, value in policy.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))
        corrupted = list(experience)
        corrupted[2] = corrupted[2] + 0.1
        with self.assertRaisesRegex(RuntimeError, "log probabilities"):
            audit_rollout(learner, tuple(corrupted))
