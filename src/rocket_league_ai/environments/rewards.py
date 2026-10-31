"""Versioned reward profiles; shaping is a hypothesis, not proven playing skill."""
import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward


def speed_toward_ball(car, ball) -> float:
    """Signed car speed in the direction of the ball, with safe zero distance."""
    delta = ball.position - car.physics.position
    distance = float(np.linalg.norm(delta))
    return float(np.dot(car.physics.linear_velocity, delta / distance)) if distance > 1e-6 else 0.0


class ApproachReward(RewardFunction):
    def __init__(self, action_repeat: int):
        self.seconds = action_repeat / 120

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info) -> dict:
        # At most 0.25 per simulated second. Standing still or moving away earns zero.
        return {agent: self.seconds * 0.25 * np.clip(
            speed_toward_ball(state.cars[agent], state.ball) / 2300, 0, 1,
        ) for agent in agents}


class DirectedTouchReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info) -> dict:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            goal = np.array([0, 5120 if car.team_num == 0 else -5120, 320])
            delta = goal - state.ball.position
            distance = float(np.linalg.norm(delta))
            speed = float(np.dot(state.ball.linear_velocity, delta / distance)) if distance > 1e-6 else 0
            rewards[agent] = float(car.ball_touches > 0) * np.clip(speed / 6000, 0, 1)
        return rewards


def build_reward(profile: str = "sparse", action_repeat: int = 8) -> CombinedReward:
    """Keep the old baseline reproducible; approach_v1 adds small dense feedback."""
    if profile == "sparse":
        return CombinedReward((TouchReward(), 0.1), (GoalReward(), 1.0))
    if profile == "approach_v1":
        return CombinedReward((ApproachReward(action_repeat), 1.0), (TouchReward(), 1.0),
                              (DirectedTouchReward(), 1.0), (GoalReward(), 10.0))
    raise ValueError(f"Unknown reward profile: {profile}")
