from src.optimizers.models.individual import Individual

from abc import ABC, abstractmethod

class FitnessFunction(ABC):
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        pass

    def evaluate_population(self, population: list[Individual]):
        for individual in population:
            self.evaluate(individual)
    