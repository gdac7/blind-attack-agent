from src.optimizers.models.individual import Individual

from dataclasses import dataclass

@dataclass
class Particle:
    curr_state: Individual
    particle_best: Individual | None = None
    velocity: int = 1

    @classmethod
    def from_individual(cls, individual: Individual) -> 'Particle':
        return cls(curr_state=individual)
    