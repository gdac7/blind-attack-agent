from src.models.individual import Individual

from abc import ABC, abstractmethod

class FitnessFunction(ABC):
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        pass
    