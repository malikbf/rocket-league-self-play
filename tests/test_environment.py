"""CPU integration check bounded by a subprocess deadline."""
import pickle
import subprocess
import sys
import unittest
from pathlib import Path

from rocket_league_ai.environments.factory import build_env


class EnvironmentTests(unittest.TestCase):
    def test_factory_is_picklable(self):
        self.assertIs(pickle.loads(pickle.dumps(build_env)), build_env)

    def test_invalid_parameters(self):
        for kwargs in (
            {"action_repeat": 0}, {"action_repeat": True}, {"action_repeat": 1.5},
            {"action_repeat": 121}, {"episode_seconds": 0},
            {"episode_seconds": float("nan")}, {"no_touch_seconds": float("inf")},
            {"no_touch_seconds": -1}, {"episode_seconds": "1"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_env(**kwargs)

    def test_bounded_rollout(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "scripts/verify_environment.py"), "--worker"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
