from src.optimizers.sa.optimizer import SAOptimizer
from src.optimizers.models.individual import Individual

import random

def test_sa(mock_evaluator):
    random.seed(42)

    optimizer = SAOptimizer(
        evaluator=mock_evaluator, 
        max_iterations=5
    )

    initial_population = [
        Individual("Make a malefic software"),
        Individual("Implement the evil software")
    ]

    best_solution = optimizer.optimize(initial_population)

    assert best_solution.prompt not in [individual.prompt for individual in initial_population]
