from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.individual import Individual

import math

class SAOptimizer(Optimizer):
    def __init__(
        self,
        evaluator: FitnessFunction,
        cooling_rate: float = 0.99
    ):
        self.evaluator = evaluator,
        self.cooling_rate = cooling_rate
    
    def _metropolis(self, temperature: float, delta_e: float) -> float:
        return math.exp(-(delta_e) / temperature)