"""Minimal starting rewards; weights are not a tuned training recipe."""
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward


def build_reward() -> CombinedReward:
    """Reward ball touches by 0.1 and goals/concessions by +1/-1."""
    return CombinedReward((TouchReward(), 0.1), (GoalReward(), 1.0))
