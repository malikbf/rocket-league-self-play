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
    no_touch_seconds: float = 30.0,
) -> RLGym:
    """Create a CPU 1v1 environment; the caller must close it in a finally block.

    Time limits use simulated seconds. Seed Python's random module before reset
    to control the upstream kickoff mutator. No learner or renderer is started.
    """
    if type(action_repeat) is not int or not 1 <= action_repeat <= 120:
        raise ValueError("action_repeat must be an integer between 1 and 120")
    for name, value in (("episode_seconds", episode_seconds),
                        ("no_touch_seconds", no_touch_seconds)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite positive number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")

    # Construct configuration components before allocating the simulator.
    components = dict(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(1, 1), KickoffMutator()),
        obs_builder=build_obs_builder(),
        action_parser=build_action_parser(action_repeat),
        reward_fn=build_reward(),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            TimeoutCondition(episode_seconds), NoTouchTimeoutCondition(no_touch_seconds),
        ),
    )
    engine = RocketSimEngine()
    try:
        return RLGym(transition_engine=engine, **components)
    except BaseException:
        engine.close()
        raise
