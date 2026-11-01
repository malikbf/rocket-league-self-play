"""Short checkpoint/inference checks plus a bounded evaluation subprocess."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from rlgym_ppo.ppo import DiscreteFF

from rocket_league_ai.agents.policies import choose_action, load_policy
from rocket_league_ai.training.checkpoints import restore_checkpoint
from rocket_league_ai.training.config import SmokeConfig


class CheckpointTests(unittest.TestCase):
    def test_inference_roundtrip_preserves_rng(self):
        torch.manual_seed(42)
        policy = DiscreteFF(92, 90, (32, 32), "cpu")
        obs = np.linspace(-1, 1, 92, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save(policy.state_dict(), path)
            rng = torch.get_rng_state().clone()
            restored = load_policy(path)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            with torch.no_grad():
                self.assertTrue(torch.equal(policy.get_output(obs), restored.get_output(obs)))
            action = choose_action(restored, obs)
            self.assertEqual(action.shape, (1,))
            self.assertTrue(0 <= action[0] < 90)
            self.assertFalse(restored.training)

    def test_nonfinite_checkpoint_rejected(self):
        policy = DiscreteFF(92, 90, (32, 32), "cpu")
        with torch.no_grad():
            next(policy.parameters()).fill_(float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save(policy.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_policy(path)

    def test_incompatible_manifest_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({"format": 999}))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                restore_checkpoint(SimpleNamespace(), path, SmokeConfig())

    def test_evaluation_repeatable_and_does_not_change_policy(self):
        code = '''
import random
import numpy as np
import torch
from rlgym_ppo.ppo import DiscreteFF
from rocket_league_ai.evaluation.runner import evaluate
torch.set_num_threads(1)
torch.manual_seed(42)
policy = DiscreteFF(92, 90, (32, 32), "cpu")
policy.eval()
before = {key: value.clone() for key, value in policy.state_dict().items()}
random.seed(50)
rng = random.getstate()
first = evaluate(policy, policy)
second = evaluate(policy, policy)
assert first == second
assert random.getstate() == rng
assert first["wins"] + first["losses"] + first["draws"] == 4
assert len(first["episodes"]) == 4
assert all(1 <= row["steps"] <= 150 for row in first["episodes"])
assert all(torch.equal(before[key], value) for key, value in policy.state_dict().items())
custom = evaluate(policy, policy, seeds=(1234,))
assert custom["seeds"] == [1234] and len(custom["episodes"]) == 2
for seeds in ((), (1, 1), (-1,), (True,), tuple(range(21)), [1]):
 try:
  evaluate(policy, policy, seeds=seeds)
  raise AssertionError("invalid seeds accepted")
 except ValueError:
  pass
torch_rng = torch.get_rng_state().clone()
numpy_rng = np.random.get_state()
sampled = evaluate(policy, policy, start_mode="approach", action_selection="sample")
assert sampled == evaluate(policy, policy, start_mode="approach", action_selection="sample")
assert torch.equal(torch_rng, torch.get_rng_state())
assert all(np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state()))
assert random.getstate() == rng
assert sampled["action_selection"] == "sample"
assert sampled["opponent_action_selection"] == "argmax"
assert 0 <= sampled["touch_episode_rate"] <= 1
for row in sampled["episodes"]:
 assert sum(row["action_counts"]) == row["steps"]
 assert 0 <= row["mean_action_entropy_nats"] <= np.log(90) + 1e-6
 assert 0 < row["mean_max_action_probability"] <= 1
 assert np.isfinite(row["closest_ball_distance"]) and row["closest_ball_distance"] >= 0
try:
 evaluate(policy, policy, action_selection="invalid")
 raise AssertionError("invalid mode accepted")
except ValueError:
 pass
assert all(torch.equal(before[key], value) for key, value in policy.state_dict().items())
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
