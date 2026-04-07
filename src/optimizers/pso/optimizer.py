from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.particle import Particle
from src.optimizers.models.individual import Individual

import copy

class PSOOptimizer(Optimizer):
    def __init__(
        self,
        evaluator: FitnessFunction
    ):
        super().__init__('Particle Swarm Optimization')

        self.evaluator = evaluator
    
    def _init_swarm(self, initial_population: list[Individual]) -> list[Particle]:
        return [Particle.from_individual(individual) for individual in Individual]
    
    def _find_gbest(self, swarm: list[Particle]) -> Individual:
        gbest = swarm[0].curr_state
        self.evaluator.evaluate(gbest)

        for particle in swarm:
            individual = particle.curr_state
            self.evaluator.evaluate(individual)

            if individual.fitness > gbest.fitness:
                gbest = individual
            
        return gbest

    def _evaluate_curr(self, particle: Particle) -> float:
        self.evaluator.evaluate(particle.curr_state)

        if (not particle.particle_best) or (particle.curr_state.fitness > particle.particle_best.fitness):
            particle.particle_best = copy.deepcopy(particle.curr_state)

        return particle.curr_state.fitness
