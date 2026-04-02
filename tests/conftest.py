from src.optimizers.fitness.base import FitnessFunction

import pytest

class EvaluatorMock(FitnessFunction):
    def evaluate(self, prompt: str) -> float:
        words = prompt.split()
        
        if not words:
            return 0.0
            
        total_letters = sum(len(word) for word in words)
        return float(total_letters / len(words))
    
@pytest.fixture
def mock_evaluator():
    return EvaluatorMock()
