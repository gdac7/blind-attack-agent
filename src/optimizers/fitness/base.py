from src.optimizers.models.individual import Individual

from abc import ABC, abstractmethod

class FitnessFunction(ABC):
    @abstractmethod
    def evaluate(self, prompt: str) -> float:
        pass
    