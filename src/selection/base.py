from abc import ABC, abstractmethod

class Selection(ABC):
    @abstractmethod
    def select(self, offsprings: list, pop_size: int):
        pass
