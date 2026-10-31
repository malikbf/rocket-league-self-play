"""JSON-compatible simulator snapshots shared with the isolated viewer."""
from typing import Any
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BOOST_LOCATIONS


def snapshot(state: GameState, fps: float) -> dict[str, Any]:
    """Serialize visual state only; no policies or training data are exposed."""
    def physics(obj):
        return {"position": obj.position.tolist(), "velocity": obj.linear_velocity.tolist(),
                "angular_velocity": obj.angular_velocity.tolist(),
                "rotation": obj.rotation_mtx.transpose().flatten().tolist()}

    return {"fps": fps, "boost_locations": [list(p) for p in BOOST_LOCATIONS],
            "frame": {"ball": physics(state.ball),
                      "pads": [bool(timer == 0) for timer in state.boost_pad_timers],
                      "cars": [{"id": index + 1, "team": car.team_num,
                                "hitbox": car.hitbox_type, "boost": float(car.boost_amount),
                                "physics": physics(car.physics)}
                               for index, car in enumerate(state.cars.values())]}}
