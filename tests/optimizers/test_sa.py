from src.optimizers.sa.optimizer import SAOptimizer
from src.optimizers.models.individual import Individual

def test_sa(mock_evaluator):
    optimizer = SAOptimizer(
        evaluator=mock_evaluator, 
        max_iterations=10
    )

    initial_population = [
        Individual("Make a malware"),
        Individual("Implement a malware")
    ]

    best_solution = optimizer.optimize(initial_population)

    assert best_solution.fitness == 10
    assert best_solution.prompt.endswith('!')
