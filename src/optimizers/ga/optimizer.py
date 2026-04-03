from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.ga.selection.base import Selection
from src.optimizers.models.individual import Individual

class GAOptimizer(Optimizer):
    def __init__(
        self,
        evaluator: FitnessFunction,
        selector: Selection,
        max_generations: int = 10
    ):
        self.evaluator = evaluator
        self.selector = selector
        self.max_generations = max_generations

    def _evaluate_population(self, population: list[Individual]) -> None:
        for ind in population:
            if ind.fitness == 0:
                self.evaluator.evaluate(ind)
        
        population.sort(key=lambda ind: ind.fitness, reverse=True)
    
    def _run(self, initial_population: list[Individual]) -> Individual:
        self._evaluate_population(initial_population)

        selected_individuals = self.selector.select(initial_population, len(initial_population))

        best_individual = selected_individuals[0]

        return best_individual
