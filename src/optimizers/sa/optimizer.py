from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.individual import Individual

import math
import random

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
    
    def _anneal_individual(self, initial_solution: Individual) -> Individual:
        temp = self._calibrate_temperature(initial_solution)

        curr_solution = initial_solution
        curr_fitness = self.evaluator.evaluate(initial_solution)

        for i in range(self.max_iterations):
            neighbor = self._generate_neighbor(curr_solution)
            neighbor_fitness = neighbor.fitness

            if neighbor_fitness < curr_fitness:
                curr_solution = neighbor
                curr_fitness = neighbor_fitness
            else:
                delta_e = neighbor_fitness - curr_fitness
                metropolis_prob = self._metropolis(temp, delta_e)

                if random.random() < metropolis_prob:
                    curr_solution = neighbor
                    curr_fitness = neighbor_fitness
            
            temp = temp * self.cooling_rate

        return curr_solution
