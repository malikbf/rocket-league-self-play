"""Optional real-time state publishing for a single simulator worker."""
import json
from pathlib import Path
import time

from rlgym_ppo.util import RLGymV2GymWrapper

from rocket_league_ai.environments.frames import snapshot


class LiveEnv(RLGymV2GymWrapper):
    """Publish atomic latest-state files and pace actual training decisions."""
    def __init__(self, env, path: Path, action_repeat: int):
        self.path = path
        self.interval = action_repeat / 120
        self.sequence = 0
        self.last_step = time.monotonic()
        super().__init__(env)

    def _publish(self) -> None:
        data = snapshot(self.rlgym_env.state, 1 / self.interval)
        data["sequence"] = self.sequence
        self.sequence += 1
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
        temporary.replace(self.path)

    def reset(self):
        obs = super().reset()
        self._publish()
        self.last_step = time.monotonic()
        return obs

    def step(self, actions):
        time.sleep(max(0, self.last_step + self.interval - time.monotonic()))
        result = super().step(actions)
        self._publish()
        self.last_step = time.monotonic()
        return result
