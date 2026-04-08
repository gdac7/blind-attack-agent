from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.particle import Particle
from src.optimizers.models.individual import Individual
from src.optimizers.utils.nlp_constants import TARGET_POS_TAGS

import copy
import random

import spacy
from loguru import logger

class PSOOptimizer(Optimizer):
    DEFAULT_C1 = 1.496
    DEFAULT_C2 = 1.496
    DEFAULT_WMAX = 0.9
    DEFAULT_WMIN = 0.4
    DEFAULT_MAX_ITER = 20

    def __init__(
        self,
        evaluator: FitnessFunction,
        max_iter: int = DEFAULT_MAX_ITER,
        c1: float = DEFAULT_C1,
        c2: float = DEFAULT_C2,
        wmax: float = DEFAULT_WMAX,
        wmin: float = DEFAULT_WMIN
    ):
        super().__init__('Particle Swarm Optimization')

        self.nlp = spacy.load('en_core_web_sm')

        self.evaluator = evaluator
        self.max_iter = max_iter

        self.c1 = c1
        self.c2 = c2
        self.wmax = wmax
        self.wmin = wmin
    
    def _init_swarm(self, initial_population: list[Individual]) -> list[Particle]:
        return [Particle.from_individual(individual) for individual in initial_population]
    
    def _find_gbest(self, swarm: list[Particle]) -> Individual:
        gbest = swarm[0].curr_state
        self.evaluator.evaluate(gbest)

        for particle in swarm:
            individual = particle.curr_state
            self.evaluator.evaluate(individual)

            if individual.fitness > gbest.fitness:
                gbest = individual
            
        return gbest
    
    def _prompt_distance(self, curr_prompt: str, target_prompt: str) -> int:
        curr_doc = self.nlp(curr_prompt)
        target_doc = self.nlp(target_prompt)

        curr_valid = set([token.lemma_ for token in curr_doc if token.pos_ in TARGET_POS_TAGS])
        target_valid = set([token.lemma_ for token in target_doc if token.pos_ in TARGET_POS_TAGS])

        sets_diff = target_valid - curr_valid

        return len(sets_diff)
    
    def _calc_inertia(self, iter: int) -> float:
        return self.wmax - ((self.wmax - self.wmin) / self.max_iter) * iter
    
    def _calc_velocity(self, particle: Particle, gbest: Individual, iter: int) -> float:
        w = self._calc_inertia(iter)
        r1 = random.random()
        r2 = random.random()

        dist_pbest = self._prompt_distance(particle.curr_state.prompt, gbest.prompt)
        dist_gbest = self._prompt_distance(particle.curr_state.prompt, particle.pbest.prompt)

        velocity = round((w * particle.velocity) + (self.c1 * r1 * dist_pbest) + (self.c2 * r2 * dist_gbest))

        particle.velocity = velocity

        return velocity

    def _evaluate_curr(self, particle: Particle) -> float:
        self.evaluator.evaluate(particle.curr_state)

        if (not particle.pbest) or (particle.curr_state.fitness > particle.pbest.fitness):
            particle.pbest = copy.deepcopy(particle.curr_state)

        return particle.curr_state.fitness
    
    def _run(self, initial_population: list[Individual]) -> Individual:
        swarm = self._init_swarm(initial_population)

        gbest = self._find_gbest(swarm)

        for particle in swarm:
            curr_fitness = self._evaluate_curr(particle)

            if curr_fitness > gbest.fitness:
                gbest = particle.curr_state
