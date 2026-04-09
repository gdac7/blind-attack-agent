from src.optimizers.models.individual import Individual

import copy
from dataclasses import dataclass

@dataclass
class Particle:
    curr_state: Individual
    pbest: Individual | None = None
    velocity: int = 1

    @classmethod
    def from_individual(cls, individual: Individual) -> 'Particle':
        return cls(curr_state=individual)
    
    def update_pbest(self) -> None:
        if (not self.pbest) or (self.curr_state.fitness > self.pbest.fitness):
            self.pbest = copy.deepcopy(self.curr_state)     
    