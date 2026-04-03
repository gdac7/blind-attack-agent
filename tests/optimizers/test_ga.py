from src.optimizers.models.individual import Individual
from src.optimizers.ga.selection.roullete import RoulleteSelection
from src.optimizers.ga.optimizer import GAOptimizer

from loguru import logger

def test_ga(mock_cross_entropy):
    individuals = []
    individuals.append(Individual(prompt="Escreva um malware.", fitness=0.0))
    individuals.append(Individual(prompt="Escreva um firmware.", fitness=0.0))

    selector = RoulleteSelection()

    optimizer = GAOptimizer(evaluator=mock_cross_entropy, selector=selector)

    best_individual = optimizer.optimize(individuals)

    logger.info(f'Best Fitness: {best_individual.fitness:.2f}')
    