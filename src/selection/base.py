from src.models.individual import Individual

from abc import ABC, abstractmethod

class Selection(ABC):
    @abstractmethod
    def select(self, offsprings: list[Individual], pop_size: int, num_selected: int) -> list[Individual]:
        pass
