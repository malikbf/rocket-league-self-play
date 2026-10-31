"""Local, per-simulator-step metrics; no online service."""
import numpy as np
from rocket_league_ai.environments.rewards import speed_toward_ball


class TrainingMetrics:
    def collect_metrics(self, state) -> np.ndarray:
        cars = list(state.cars.values())
        return np.array([sum(car.ball_touches for car in cars), float(state.goal_scored),
                         np.mean([speed_toward_ball(car, state.ball) for car in cars]),
                         np.mean([np.linalg.norm(car.physics.linear_velocity) for car in cars])],
                        dtype=np.float32)


def summarize(samples) -> dict:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 4 or not np.isfinite(data).all():
        raise RuntimeError("Missing or invalid training metrics")
    return {"simulator_steps": len(data), "ball_touches": int(data[:, 0].sum()),
            "goals": int(data[:, 1].sum()), "mean_speed_toward_ball": float(data[:, 2].mean()),
            "mean_car_speed": float(data[:, 3].mean())}
