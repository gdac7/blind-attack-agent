from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction

class SAOptimizer(Optimizer):
    def __init__(
        self,
        evaluator: FitnessFunction,
        cooling_rate: float = 0.99
    ):
        self.evaluator = evaluator,
        self.cooling_rate = cooling_rate