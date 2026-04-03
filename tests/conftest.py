from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.individual import Individual
from src.optimizers.fitness.cross_entropy import CrossEntropyFitness

import pytest
import torch
from unittest.mock import MagicMock

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

@pytest.fixture
def mock_cross_entropy():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_tokenizer.encode.side_effect = lambda text, **kwargs: [99, 100, 101] if "Sure" in text else [1, 2, 3, 4]

    mock_model.device = torch.device("cpu")

    logits = torch.zeros((1, 7, 50000))
    logits[0, 3, 99] = 20.0
    logits[0, 4, 100] = 20.0
    logits[0, 5, 101] = 20.0

    mock_outputs = MagicMock()
    mock_outputs.logits = logits
    mock_model.return_value = mock_outputs

    return CrossEntropyFitness(
        model=mock_model,
        tokenizer=mock_tokenizer
    )
