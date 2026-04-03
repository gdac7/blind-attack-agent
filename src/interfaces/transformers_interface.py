from abc import ABC, abstractmethod

class TransformersModel(ABC):
    @abstractmethod
    def generate(self):
        pass