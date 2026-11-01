"""Driving geometry, discrete controls and bounded contact calibration."""
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction

from rocket_league_ai.agents.baselines import DriveToBallPolicy


class BaselineTests(unittest.TestCase):
    def test_ground_controls_and_rotation_invariance(self):
        table = LookupTableAction.make_lookup_table()
        policy = DriveToBallPolicy(table)
        for target, steer, boost in (((1, 0), 0, 1), ((1, 1), 1, 0), ((1, -1), -1, 0)):
            obs = np.zeros(92)
            obs[:2] = target
            obs[55] = 1
            expected = np.array([1, steer, 0, steer, 0, 0, boost, 0])
            probabilities = policy.get_output(obs)
            self.assertEqual(float(probabilities.sum()), 1)
            np.testing.assert_array_equal(table[int(probabilities.argmax())], expected)
            # Inversion used for the opposite team must not reverse steering.
            obs[:2] *= -1
            obs[55:57] *= -1
            np.testing.assert_array_equal(policy.get_output(obs).numpy(), probabilities.numpy())
        obs = np.zeros(92)
        obs[0], obs[55] = 0.1, 1
        self.assertEqual(table[int(policy.get_output(obs).argmax()), 6], 0)
        with self.assertRaises(ValueError):
            policy.get_output(np.full(92, np.nan))

    def test_bounded_contact_on_both_teams(self):
        code = '''
import torch
from rlgym_ppo.ppo import DiscreteFF
from rlgym.rocket_league.action_parsers import LookupTableAction
from rocket_league_ai.agents.baselines import DriveToBallPolicy
from rocket_league_ai.evaluation.runner import evaluate
torch.set_num_threads(1)
torch.manual_seed(42)
opponent = DiscreteFF(92, 90, (32,32), "cpu")
policy = DriveToBallPolicy(LookupTableAction.make_lookup_table())
for start in ("kickoff", "approach"):
 report = evaluate(policy, opponent, start_mode=start, seeds=(1001,))
 assert report["touch_episode_rate"] == 1, report
 assert all(row["touches"] > 0 and row["steps"] <= 150 for row in report["episodes"])
 assert all(row["mean_action_entropy_nats"] == 0 for row in report["episodes"])
'''
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
