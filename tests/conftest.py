from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.individual import Individual

import pytest

class EvaluatorMock(FitnessFunction):
    def evaluate(self, individual: Individual) -> float:
        words = individual.prompt.split()
        
        if not words:
            return 0.0
            
        total_letters = sum(len(word) for word in words)
        fitness = float(total_letters / len(words))

        individual.fitness = fitness
        
        return fitness
    
@pytest.fixture
def mock_evaluator():
    return EvaluatorMock()
