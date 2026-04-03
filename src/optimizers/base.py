from src.optimizers.models.individual import Individual

import time
from loguru import logger
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, optimizer_name: str):
        self.optimizer_name = optimizer_name

    def optimize(self, initial_population: list[Individual]) -> Individual:
        start_time = time.time()

        best_individual = self._run(initial_population)

        total_time = time.time() - start_time
        logger.info(f'{self.optimizer_name} | Total Time: {total_time:.2f}')

        return best_individual
    
    @abstractmethod
    def _run(self, initial_population: list[Individual]) -> Individual:
        pass
