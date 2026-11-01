"""Bounded fixed-seed evaluation against a frozen initial policy."""
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rocket_league_ai.agents.policies import ActionPolicy


def evaluate(policy: "ActionPolicy", opponent: "ActionPolicy", *, start_mode: str = "kickoff",
             action_selection: str = "argmax", seeds: tuple[int, ...] = (1001, 1002)) -> dict:
    """Swap sides for each seed, at most 150 steps each (four episodes by default).

    No gradients or optimization. The frozen opponent and evaluation protocol
    must stay the same when comparing checkpoints. These short games are only
    an initial evaluation baseline, not a competitive skill benchmark.
    """
    from rocket_league_ai.agents.policies import choose_action
    from rocket_league_ai.environments.factory import build_env
    import torch

    if action_selection not in ("argmax", "sample"):
        raise ValueError("action_selection must be argmax or sample")
    if (not isinstance(seeds, tuple) or not 1 <= len(seeds) <= 20
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a tuple of 1–20 distinct uint32 integers")

    records = []
    random_state = random.getstate()
    try:
        for seed in seeds:
            for team in ("blue", "orange"):
                random.seed(seed)
                # Local generator: evaluation must not advance training RNG streams.
                rng = np.random.default_rng(seed)
                env = build_env(episode_seconds=10.0, no_touch_seconds=0, start_mode=start_mode)
                agent = f"{team}-0"
                other = "orange-0" if team == "blue" else "blue-0"
                total_reward, touches = 0.0, 0
                outcome = "draw"
                try:
                    obs = env.reset()
                    closest = float(np.linalg.norm(env.state.ball.position - env.state.cars[agent].physics.position))
                    counts = np.zeros(90, dtype=np.int64)
                    entropy_sum, maximum_sum = 0.0, 0.0
                    for step in range(1, 151):
                        with torch.no_grad():
                            probabilities = policy.get_output(obs[agent]).detach().cpu().numpy().astype(np.float64)
                        if (probabilities.shape != (90,) or not np.isfinite(probabilities).all()
                                or (probabilities < 0).any() or probabilities.sum() <= 0):
                            raise ValueError("Expected finite nonnegative action probabilities")
                        probabilities /= probabilities.sum()
                        action = int(probabilities.argmax()) if action_selection == "argmax" else int(rng.choice(90, p=probabilities))
                        counts[action] += 1
                        positive = probabilities[probabilities > 0]
                        entropy_sum += float(-(positive * np.log(positive)).sum())
                        maximum_sum += float(probabilities.max())
                        obs, rewards, terminated, truncated = env.step({
                            agent: np.array([action], dtype=np.int64),
                            other: choose_action(opponent, obs[other]),
                        })
                        if not all(np.isfinite(value).all() for value in obs.values()):
                            raise RuntimeError("Nonfinite evaluation observations")
                        total_reward += float(rewards[agent])
                        touches += int(env.state.cars[agent].ball_touches)
                        closest = min(closest, float(np.linalg.norm(
                            env.state.ball.position - env.state.cars[agent].physics.position)))
                        if any(terminated.values()):
                            outcome = "win" if env.state.scoring_team == env.state.cars[agent].team_num else "loss"
                            break
                        if any(truncated.values()):
                            break
                    records.append({"seed": seed, "team": team, "outcome": outcome,
                                    "steps": step, "reward": total_reward, "touches": touches,
                                    "closest_ball_distance": closest,
                                    "action_counts": counts.tolist(),
                                    "mean_action_entropy_nats": entropy_sum / step,
                                    "mean_max_action_probability": maximum_sum / step})
                finally:
                    env.close()
    finally:
        random.setstate(random_state)
    return {
        "protocol": "frozen-initial-v1" if start_mode == "kickoff" else "approach-practice-v1", "seeds": list(seeds),
        "action_selection": action_selection, "max_steps_per_episode": 150,
        "evaluated_policy_action_selection": action_selection,
        "opponent_action_selection": "argmax",
        "wins": sum(row["outcome"] == "win" for row in records),
        "losses": sum(row["outcome"] == "loss" for row in records),
        "draws": sum(row["outcome"] == "draw" for row in records),
        "mean_reward": sum(row["reward"] for row in records) / len(records),
        "touches": sum(row["touches"] for row in records),
        "touch_episode_rate": sum(row["touches"] > 0 for row in records) / len(records),
        "mean_closest_ball_distance": sum(row["closest_ball_distance"] for row in records) / len(records),
        "episodes": records,
    }
