"""Bounded CPU environment smoke check, with a hard subprocess timeout."""
import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from rocket_league_ai.environments.factory import build_env


def check_observations(env, observations: dict) -> None:
    """Validate agent coverage, declared shape, and finite observations."""
    assert len(observations) == 2
    assert set(observations) == set(env.agents)
    for agent, obs in observations.items():
        assert obs.shape == (env.observation_space(agent)[1],) == (92,)
        assert np.isfinite(obs).all()
        assert env.action_space(agent) == ("discrete", 90)


def run_checks() -> None:
    """Run 120 random steps, independent timeout checks, and a forced goal."""
    random.seed(42)
    rng = np.random.default_rng(42)
    env = build_env(episode_seconds=1.0, no_touch_seconds=30.0)
    try:
        check_observations(env, env.reset())
        resets = 0
        for _ in range(120):
            actions = {agent: np.array([rng.integers(90)]) for agent in env.agents}
            obs, rewards, terminated, truncated = env.step(actions)
            check_observations(env, obs)
            for values in (rewards, terminated, truncated):
                assert set(values) == set(obs)
            assert all(np.isfinite(reward) for reward in rewards.values())
            if any(terminated.values()) or any(truncated.values()):
                check_observations(env, env.reset())
                resets += 1
        assert resets >= 8, "Episode time limit did not trigger"
    finally:
        env.close()

    # Stationary controls isolate the two timeout conditions from ball touches.
    for episode_seconds, no_touch_seconds in ((0.2, 30.0), (30.0, 0.2)):
        env = build_env(episode_seconds=episode_seconds, no_touch_seconds=no_touch_seconds)
        try:
            env.reset()
            for step in range(3):
                _, _, terminated, truncated = env.step(
                    {agent: np.array([8]) for agent in env.agents}
                )
                assert not any(terminated.values())
                assert all(truncated.values()) == (step == 2)
            env.reset()
            _, _, _, truncated = env.step({agent: np.array([8]) for agent in env.agents})
            assert not any(truncated.values()), "Timeout state did not reset"
        finally:
            env.close()

    env = build_env()
    try:
        env.reset()
        state = env.state
        state.ball.position = np.array([0, 5200, 100], dtype=np.float32)
        state.ball.linear_velocity = np.array([0, 1000, 0], dtype=np.float32)
        env.set_state(state)
        _, rewards, terminated, truncated = env.step(
            {agent: np.array([8]) for agent in env.agents}
        )
        assert all(terminated.values()) and not any(truncated.values())
        assert rewards == {"blue-0": 1.0, "orange-0": -1.0}
        check_observations(env, env.reset())
        assert not env.state.goal_scored
    finally:
        env.close()
    print("PASS: 120 rollout steps, finite 92-value observations, 90 actions, "
          "episode/inactivity limits, goal rewards, reset and close; seed=42")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_checks()
        return 0
    try:
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            timeout=30, check=False,
        ).returncode
    except subprocess.TimeoutExpired:
        print("FAIL: environment check exceeded 30 seconds; worker killed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
