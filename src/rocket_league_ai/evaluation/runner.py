"""Bounded fixed-seed evaluation against a frozen initial policy."""
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rlgym_ppo.ppo import DiscreteFF


def evaluate(policy: "DiscreteFF", opponent: "DiscreteFF") -> dict:
    """Play four episodes, swapping sides for each seed, at most 150 steps each.

    No gradients or optimization. The frozen opponent and evaluation protocol
    must stay the same when comparing checkpoints. These short games are only
    an initial evaluation baseline, not a competitive skill benchmark.
    """
    from rocket_league_ai.agents.policies import choose_action
    from rocket_league_ai.environments.factory import build_env

    records = []
    random_state = random.getstate()
    try:
        for seed in (1001, 1002):
            for team in ("blue", "orange"):
                random.seed(seed)
                env = build_env(episode_seconds=10.0, no_touch_seconds=10.0)
                agent = f"{team}-0"
                other = "orange-0" if team == "blue" else "blue-0"
                total_reward, touches = 0.0, 0
                outcome = "draw"
                try:
                    obs = env.reset()
                    for step in range(1, 151):
                        obs, rewards, terminated, truncated = env.step({
                            agent: choose_action(policy, obs[agent]),
                            other: choose_action(opponent, obs[other]),
                        })
                        if not all(np.isfinite(value).all() for value in obs.values()):
                            raise RuntimeError("Nonfinite evaluation observations")
                        total_reward += float(rewards[agent])
                        touches += int(env.state.cars[agent].ball_touches)
                        if any(terminated.values()):
                            outcome = "win" if env.state.scoring_team == env.state.cars[agent].team_num else "loss"
                            break
                        if any(truncated.values()):
                            break
                    records.append({"seed": seed, "team": team, "outcome": outcome,
                                    "steps": step, "reward": total_reward, "touches": touches})
                finally:
                    env.close()
    finally:
        random.setstate(random_state)
    return {
        "protocol": "frozen-initial-v1", "seeds": [1001, 1002],
        "action_selection": "argmax", "max_steps_per_episode": 150,
        "wins": sum(row["outcome"] == "win" for row in records),
        "losses": sum(row["outcome"] == "loss" for row in records),
        "draws": sum(row["outcome"] == "draw" for row in records),
        "mean_reward": sum(row["reward"] for row in records) / len(records),
        "touches": sum(row["touches"] for row in records), "episodes": records,
    }
