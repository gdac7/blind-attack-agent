from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.particle import Particle

import copy

class PSOOptimizer(Optimizer):
    def __init__(
        self,
        evaluator: FitnessFunction
    ):
        super().__init__('Particle Swarm Optimization')

        self.evaluator = evaluator

    def _evaluate_curr(self, particle: Particle) -> float:
        self.evaluator.evaluate(particle.curr_state)

        if (not particle.particle_best) or (particle.curr_state.fitness > particle.particle_best.fitness):
            particle.particle_best = copy.deepcopy(particle.curr_state)

        return particle.curr_state.fitness
