from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.individual import Individual

import math

class SAOptimizer(Optimizer):
    DEFAULT_INITIAL_TEMP = 100
    DEFAULT_COOLING_RATE = 0.99
    DEFAULT_MAX_ITER     = 20

    def __init__(
        self,
        evaluator: FitnessFunction,
        cooling_rate: float = DEFAULT_COOLING_RATE,
        max_iterations: int = DEFAULT_MAX_ITER
    ):
        self.evaluator = evaluator
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
    
    def _metropolis(self, temperature: float, delta_e: float) -> float:
        return math.exp(-(delta_e) / temperature)
    
    def _calibrate_temperature(self, initial_solution: Individual) -> float:
        return self.DEFAULT_INITIAL_TEMP
    
    def _generate_neighbor(self, current_solution: Individual) -> Individual:
        neighbor_prompt = current_solution.prompt + '!'
        neighbor = Individual(neighbor_prompt)

        return neighbor