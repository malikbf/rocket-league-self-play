"""Short config/adapter tests; the actual PPO smoke run is an explicit command."""
from functools import partial
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from rocket_league_ai.training.config import EnvironmentConfig, load_config
from rocket_league_ai.training.ppo import build_ppo_env

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def load_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_baseline(self):
        config = load_config(ROOT / "configs/baseline.toml")
        self.assertEqual(config.training.timesteps, 256)
        self.assertEqual(config.training.device, "cpu")

    def test_defaults(self):
        self.assertEqual(self.load_text("").environment.action_repeat, 8)

    def test_reject_bad_config(self):
        for text in (
            "[unknown]", "[training]\nbacth_size=64", "training=4",
            '[training]\ndevice="mps"', '[experiment]\nname="../escape"',
            "[experiment]\nseed=-1", "[training]\ntimesteps=true",
            "[training]\ntimesteps=1000000", "[training]\ntimesteps=255",
            "[training]\nbatch_size=512", "[training]\nminibatch_size=30",
            "[training]\nepochs=0", "[training]\niterations=11", "[training]\ntimeout_seconds=301",
            "[environment]\naction_repeat=0", "[environment]\nepisode_seconds=nan",
            "[environment]\nno_touch_seconds=-1",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load_text(text)

    def test_ppo_factory_pickling(self):
        factory = partial(build_ppo_env, EnvironmentConfig(), 42)
        restored = pickle.loads(pickle.dumps(factory))
        self.assertIs(restored.func, build_ppo_env)
        self.assertEqual(restored.args, factory.args)

    def test_adapter_rollout(self):
        # Isolate native physics/import failures with a wall-clock limit.
        code = '''
import numpy as np
from rocket_league_ai.training.ppo import build_ppo_env
from rocket_league_ai.training.config import EnvironmentConfig
env = build_ppo_env(EnvironmentConfig(episode_seconds=0.2), 42)
try:
    obs = env.reset()
    assert obs.shape == (2, 92)
    assert env.action_space.n == 90
    for step in range(3):
        obs, rewards, done, truncated, info = env.step(np.array([[8], [8]]))
        assert obs.shape == (2, 92) and np.isfinite(obs).all()
        assert len(rewards) == 2 and np.isfinite(rewards).all()
        assert not done and truncated == (step == 2)
    assert env.reset().shape == (2, 92)
finally:
    env.close()
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class SupervisorTests(unittest.TestCase):
    def test_timeout_stops_process_tree_and_fails(self):
        spec = importlib.util.spec_from_file_location("train_smoke", ROOT / "scripts/train_smoke.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        process = Mock()
        process.wait.side_effect = subprocess.TimeoutExpired("smoke", 120)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(module, "ROOT", Path(directory)), \
             patch.object(module.subprocess, "Popen", return_value=process), \
             patch.object(module, "stop_process_tree") as stop, \
             patch.object(sys, "argv", ["train_smoke", "--config", str(ROOT / "configs/baseline.toml")]):
            self.assertEqual(module.main(), 1)
            stop.assert_called_with(process)
            process.wait.assert_called_once_with(timeout=120)
