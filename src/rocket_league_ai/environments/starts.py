"""Optional early curriculum: start both cars near the ball, facing toward it."""
import random
import numpy as np
from rlgym.api import StateMutator


class ApproachStart(StateMutator):
    def apply(self, state, shared_info) -> None:
        # Applied after the stock kickoff mutator so all physics fields are initialized.
        state.ball.position[:2] = [random.uniform(-700, 700), random.uniform(-700, 700)]
        for car in state.cars.values():
            side = -1 if car.team_num == 0 else 1
            car.physics.position = state.ball.position + np.array(
                [random.uniform(-250, 250), side * random.uniform(900, 1500), 0], dtype=np.float32,
            )
            car.physics.position[2] = 17
            delta = state.ball.position - car.physics.position
            car.physics.euler_angles = np.array(
                [0, np.arctan2(delta[1], delta[0]) + random.uniform(-0.2, 0.2), 0], dtype=np.float32,
            )
