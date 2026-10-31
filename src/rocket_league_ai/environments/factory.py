"""Module-level, picklable factory for the initial 1v1 RocketSim environment."""
import math

from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import (
    AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition,
)
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator, KickoffMutator, MutatorSequence,
)

from rocket_league_ai.environments.actions import build_action_parser
from rocket_league_ai.environments.observations import build_obs_builder
from rocket_league_ai.environments.rewards import build_reward


def build_env(
    *, action_repeat: int = 8, episode_seconds: float = 300.0,
    no_touch_seconds: float = 30.0, reward_profile: str = "sparse", start_mode: str = "kickoff",
) -> RLGym:
    """Create a CPU 1v1 environment; the caller must close it in a finally block.

    Time limits use simulated seconds; no_touch_seconds=0 disables inactivity resets. Seed Python's random module before reset
    to control the upstream kickoff mutator. No learner or renderer is started.
    """
    if type(action_repeat) is not int or not 1 <= action_repeat <= 120:
        raise ValueError("action_repeat must be an integer between 1 and 120")
    for name, value in (("episode_seconds", episode_seconds),
                        ("no_touch_seconds", no_touch_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite positive number")
        if not math.isfinite(value) or value < 0 or (name == "episode_seconds" and value == 0):
            raise ValueError(f"{name} must be a finite positive number")

    if start_mode not in ("kickoff", "approach"):
        raise ValueError("start_mode must be kickoff or approach")
    mutators = [FixedTeamSizeMutator(1, 1), KickoffMutator()]
    if start_mode == "approach":
        from rocket_league_ai.environments.starts import ApproachStart
        mutators.append(ApproachStart())
    limits = [TimeoutCondition(episode_seconds)]
    if no_touch_seconds > 0:
        limits.append(NoTouchTimeoutCondition(no_touch_seconds))

    # Construct configuration components before allocating the simulator.
    components = dict(
        state_mutator=MutatorSequence(*mutators),
        obs_builder=build_obs_builder(),
        action_parser=build_action_parser(action_repeat),
        reward_fn=build_reward(reward_profile, action_repeat),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(*limits),
    )
    engine = RocketSimEngine()
    try:
        return RLGym(transition_engine=engine, **components)
    except BaseException:
        engine.close()
        raise
