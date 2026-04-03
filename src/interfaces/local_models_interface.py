from abc import ABC, abstractmethod

class LocalModel(ABC):
    @abstractmethod
    def generate(self):
        pass