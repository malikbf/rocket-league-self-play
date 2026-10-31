"""Reward-direction checks and bounded arena/curriculum physics checks."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from rocket_league_ai.environments.rewards import ApproachReward, DirectedTouchReward
from rocket_league_ai.training.metrics import summarize


class ApproachTests(unittest.TestCase):
    def state(self, velocity, ball_velocity=(0, 0, 0), touches=0):
        car = SimpleNamespace(team_num=0, ball_touches=touches,
                              physics=SimpleNamespace(position=np.zeros(3), linear_velocity=np.array(velocity)))
        ball = SimpleNamespace(position=np.array([0, 1000, 0]), linear_velocity=np.array(ball_velocity))
        return SimpleNamespace(cars={"car": car}, ball=ball)

    def test_approach_reward_direction_and_time_scaling(self):
        reward = ApproachReward(8)
        toward = reward.get_rewards(["car"], self.state([0, 2300, 0]), {}, {}, {})["car"]
        self.assertAlmostEqual(toward, 0.25 * 8 / 120)
        for velocity in ([0, -2300, 0], [0, 0, 0], [2300, 0, 0]):
            self.assertEqual(reward.get_rewards(["car"], self.state(velocity), {}, {}, {})["car"], 0)
        self.assertAlmostEqual(ApproachReward(4).get_rewards(
            ["car"], self.state([0, 2300, 0]), {}, {}, {})["car"], toward / 2)

    def test_directed_touch_requires_contact_and_correct_direction(self):
        reward = DirectedTouchReward()
        self.assertGreater(reward.get_rewards(["car"], self.state([0, 0, 0], [0, 1000, 0], 1), {}, {}, {})["car"], 0)
        for touches, velocity in ((0, [0, 1000, 0]), (1, [0, -1000, 0])):
            self.assertEqual(reward.get_rewards(["car"], self.state([0, 0, 0], velocity, touches), {}, {}, {})["car"], 0)

    def test_metrics_are_counted_not_inferred_from_rewards(self):
        result = summarize([[2, 0, 20, 100], [1, 1, -10, 200]])
        self.assertEqual(result["ball_touches"], 3)
        self.assertEqual(result["goals"], 1)
        self.assertEqual(result["mean_speed_toward_ball"], 5)
        with self.assertRaises(RuntimeError):
            summarize([[float("nan"), 0, 0, 0]])

    def test_enclosure_and_curriculum(self):
        code = '''
import numpy as np
from rocket_league_ai.environments.factory import build_env
env=build_env(no_touch_seconds=0, episode_seconds=2, start_mode="approach", reward_profile="approach_v1")
try:
 for pos,vel,axis,bound in [([3800,0,500],[1800,0,0],0,4096),([-3800,0,500],[-1800,0,0],0,4096),([0,0,1800],[0,0,1800],2,2044),([1500,4800,500],[0,1800,0],1,5120),([1500,-4800,500],[0,-1800,0],1,5120)]:
  env.reset(); state=env.state
  for car in state.cars.values():
   assert 850 < np.linalg.norm(car.physics.position-state.ball.position) < 1600
   assert np.dot(car.physics.forward,state.ball.position-car.physics.position)>0
  state.ball.position=np.array(pos,dtype=np.float32)
  state.ball.linear_velocity=np.array(vel,dtype=np.float32)
  env.set_state(state); bounced=False
  for _ in range(20):
   _, rewards, terminated, truncated=env.step({agent:np.array([8]) for agent in env.agents})
   assert abs(env.state.ball.position[axis]) < bound
   bounced |= env.state.ball.linear_velocity[axis] * vel[axis] < 0
   assert not any(terminated.values()) and not any(truncated.values())
   assert all(np.isfinite(r) for r in rewards.values())
  assert bounced
 # A disabled inactivity timeout must not truncate immediately; clock still ends play.
 env.reset()
 for step in range(30):
  _,_,_,truncated=env.step({agent:np.array([8]) for agent in env.agents})
  assert all(truncated.values()) == (step==29)
finally:
 env.close()
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
