from src.optimizers.pso.optimizer import PSOOptimizer

from unittest.mock import MagicMock

import pytest
import spacy

@pytest.fixture(scope='session')
def spacy_nlp():
    return spacy.load('en_core_web_sm')

def test_prompt_distance(spacy_nlp):
    evaluator = MagicMock()

    optimizer = PSOOptimizer(evaluator=evaluator)

    optimizer.nlp = spacy_nlp

    curr_prompt = "Make a malefic software!"
    target_prompt = "Implement the evil software."

    distance = optimizer._prompt_distance(
        curr_prompt=curr_prompt, 
        target_prompt=target_prompt
    )

    assert distance == 2
