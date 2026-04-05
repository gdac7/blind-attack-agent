from src.optimizers.models.individual import Individual
from src.optimizers.fitness.base import FitnessFunction

import copy
from dataclasses import dataclass

@dataclass
class Particle:
    curr_state: Individual
    particle_best: Individual | None = None
    velocity: int = 1
    
    def evaluate_curr(self, evaluator: FitnessFunction) -> float:
        evaluator.evaluate(self.curr_state)

        if (not self.particle_best) or (self.curr_state.fitness > self.particle_best.fitness):
            self.particle_best = copy.deepcopy(self.curr_state)

        return self.curr_state.fitness
    