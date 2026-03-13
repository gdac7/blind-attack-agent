from src.fitness.base import FitnessFunction
from src.models.individual import Individual

class CrossEntropyFitness(FitnessFunction):
    def evaluate(self, individual: Individual) -> float:
        return 0
    