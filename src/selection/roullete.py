from src.selection.base import Selection
from src.models.individual import Individual

import numpy as np

class RoulleteSelection(Selection):
    def select(self, offsprings: list[Individual], pop_size: int, num_selected: int = 1) -> list[Individual]:
        fitness_sum = sum([individual.fitness for individual in offsprings])
        probs = [individual.fitness/fitness_sum for individual in offsprings]

        selected = np.random.choice(
            offsprings, 
            size=num_selected, 
            replace=False, 
            p=probs
        )

        return selected