from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.individual import Individual

import math
import random

import nltk
from nltk.tokenize import word_tokenize
from loguru import logger

class SAOptimizer(Optimizer):
    DEFAULT_INITIAL_TEMP = 100
    DEFAULT_COOLING_RATE = 0.99
    DEFAULT_MAX_ITER = 20

    INITIAL_ACCEPTANCE_RATE = 0.85
    TEMP_CALIBRATION_SAMPLES = 100

    def __init__(
        self,
        evaluator: FitnessFunction,
        cooling_rate: float = DEFAULT_COOLING_RATE,
        max_iterations: int = DEFAULT_MAX_ITER
    ):
        nltk.download('punkt_tab', quiet=True)

        self.evaluator = evaluator
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
    
    def _metropolis(self, temperature: float, delta_e: float) -> float:
        return math.exp(-(delta_e) / temperature)
    
    def _calibrate_temperature(self, initial_solution: Individual) -> float:
        delta_sum = 0
        loss_count = 0

        curr_solution = initial_solution
        curr_fitness = self.evaluator.evaluate(initial_solution.prompt)

        for _ in range(self.TEMP_CALIBRATION_SAMPLES):
            neighbor = self._generate_neighbor(curr_solution)
            neighbor_fitness = self.evaluator.evaluate(neighbor.prompt)

            if neighbor_fitness < curr_fitness:
                delta_e = curr_fitness - neighbor_fitness

                delta_sum += delta_e
                loss_count += 1

                curr_solution = neighbor
                curr_fitness = neighbor_fitness
        
        if loss_count != 0:
            delta_mean = delta_sum / loss_count
            initial_temp = -(delta_mean) / math.log(self.INITIAL_ACCEPTANCE_RATE)

            return initial_temp
        
        return self.DEFAULT_INITIAL_TEMP
    
    def _generate_neighbor(self, current_solution: Individual) -> Individual:
        solution_tokens = word_tokenize(current_solution.prompt)

        neighbor_prompt = current_solution.prompt + '!'
        neighbor = Individual(neighbor_prompt)

        return neighbor
    
    def _anneal_individual(self, initial_solution: Individual) -> Individual:
        temp = self._calibrate_temperature(initial_solution)

        logger.info(f'Initial Temperature: {temp:.2f}')

        curr_solution = initial_solution
        curr_fitness = self.evaluator.evaluate(initial_solution.prompt)

        for i in range(self.max_iterations):
            neighbor = self._generate_neighbor(curr_solution)
            neighbor_fitness = self.evaluator.evaluate(neighbor.prompt)

            if neighbor_fitness > curr_fitness:
                curr_solution = neighbor
                curr_fitness = neighbor_fitness
            else:
                delta_e = curr_fitness - neighbor_fitness
                metropolis_prob = self._metropolis(temp, delta_e)

                if random.random() < metropolis_prob:
                    curr_solution = neighbor
                    curr_fitness = neighbor_fitness
            
            temp = temp * self.cooling_rate

        curr_solution.fitness = self.evaluator.evaluate(curr_solution.prompt)

        return curr_solution
    
    def _run(self, initial_population: list[Individual]) -> Individual:
        best_solution = initial_population[0]
        best_fitness = initial_population[0].fitness

        for individual in initial_population:
            solution = self._anneal_individual(individual)

            if solution.fitness > best_fitness:
                best_solution = solution
                best_fitness = solution.fitness
        
        return best_solution
