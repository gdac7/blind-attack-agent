from src.selection.base import Selection
from src.models.individual import Individual

import random

class TournamentSelection(Selection):
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, offsprings: list[Individual], pop_size: int, num_selected: int = 1) -> list[Individual]:
        selected = []

        for _ in range(num_selected):
            tournament_pool = random.sample(offsprings, self.tournament_size)

            winner = max(tournament_pool, key=lambda ind: ind.fitness)
            selected.append(winner)

        return selected