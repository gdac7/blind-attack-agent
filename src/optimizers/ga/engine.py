from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.ga.selection.base import Selection
from src.optimizers.models.individual import Individual

import time
from loguru import logger

class GAEngine:
    def __init__(
        self,
        evaluator: FitnessFunction,
        selector: Selection,
        max_generations: int = 10
    ):
        self.evaluator = evaluator
        self.selector = selector
        self.max_generations = max_generations
    
    def _evaluate_population(self, population: list[Individual]) -> None:
        for ind in population:
            if ind.fitness == 0:
                ind.fitness = self.evaluator.evaluate(ind.prompt)
        
        population.sort(key=lambda ind: ind.fitness, reverse=True)
    
    def run(self, initial_population: list[Individual]) -> Individual:
        start_time = time.time()

        self._evaluate_population(initial_population)

        selected_individuals = self.selector.select(initial_population, len(initial_population))

        best_individual = selected_individuals[0]

        total_time = time.time() - start_time
        logger.info(f'GA Total Time: {total_time:.2f}')

        return best_individual
    