"""Bounded integration check for the optional live state publisher."""
from pathlib import Path
import subprocess
import sys
import unittest


class LiveTests(unittest.TestCase):
    def test_frames_follow_actual_steps(self):
        code = '''
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from rocket_league_ai.training.config import EnvironmentConfig
from rocket_league_ai.training.ppo import build_ppo_env
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "live.json"
    env = build_ppo_env(EnvironmentConfig(), 42, path)
    try:
        env.reset()
        before = json.loads(path.read_text())
        start = time.monotonic()
        for _ in range(2):
            env.step(np.array([[8], [8]]))
        after = json.loads(path.read_text())
        assert after["sequence"] == before["sequence"] + 2
        assert after["fps"] == 15
        assert len(after["frame"]["cars"]) == 2
        assert len(after["boost_locations"]) == len(after["frame"]["pads"])
        assert time.monotonic() - start >= 0.1
        assert not path.with_suffix(".tmp").exists()
    finally:
        env.close()
'''
        result = subprocess.run([sys.executable, "-c", code], text=True,
                                capture_output=True, timeout=30,
                                cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
