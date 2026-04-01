from src.optimizers.fitness.base import FitnessFunction

import pytest

class EvaluatorMock(FitnessFunction):
    def evaluate(self, prompt: str) -> float:
        if prompt.endswith('!'):
            return 10.0
        return 5.0
    
@pytest.fixture
def mock_evaluator():
    return EvaluatorMock()
