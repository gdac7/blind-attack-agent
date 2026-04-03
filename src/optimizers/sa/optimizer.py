from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.individual import Individual
from src.optimizers.utils.nlp_constants import TARGET_POS_TAGS
from src.optimizers.utils.nlp_constants import POS_TRANSLATION
from src.optimizers.utils.validate_synonym import validate_synonym

import math
import random

import spacy
import nltk
from nltk.wsd import lesk
from loguru import logger

class SAOptimizer(Optimizer):
    DEFAULT_INITIAL_TEMP = 100
    DEFAULT_COOLING_RATE = 0.99
    DEFAULT_MAX_ITER = 20

    INITIAL_ACCEPTANCE_RATE = 0.85
    TEMP_CALIBRATION_SAMPLES = 100

    def __init__(
        self,
        evaluator: FitnessFunction,
        cooling_rate: float = DEFAULT_COOLING_RATE,
        max_iterations: int = DEFAULT_MAX_ITER
    ):
        self.nlp = spacy.load('en_core_web_sm')

        self.evaluator = evaluator
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
    
    def _metropolis(self, temperature: float, delta_e: float) -> float:
        return math.exp(-(delta_e) / temperature)
    
    def _calibrate_temperature(self, initial_solution: Individual) -> float:
        delta_sum = 0
        loss_count = 0

        curr_solution = initial_solution
        curr_fitness = self.evaluator.evaluate(initial_solution)

        for _ in range(self.TEMP_CALIBRATION_SAMPLES):
            neighbor = self._generate_neighbor(curr_solution)
            neighbor_fitness = self.evaluator.evaluate(neighbor)

            if neighbor_fitness < curr_fitness:
                delta_e = curr_fitness - neighbor_fitness

                delta_sum += delta_e
                loss_count += 1

                curr_solution = neighbor
                curr_fitness = neighbor_fitness
        
        if loss_count != 0:
            delta_mean = delta_sum / loss_count
            initial_temp = -(delta_mean) / math.log(self.INITIAL_ACCEPTANCE_RATE)

            return initial_temp
        
        return self.DEFAULT_INITIAL_TEMP
    
    def _generate_neighbor(self, curr_solution: Individual) -> Individual:
        doc = self.nlp(curr_solution.prompt)

        mutation_candidates = [token for token in doc if token.pos_ in TARGET_POS_TAGS]

        if not mutation_candidates:
            return curr_solution

        mutation_token = random.choice(mutation_candidates)
        mutation_token_pos = POS_TRANSLATION.get(mutation_token.pos_)

        synset = lesk([token.text for token in doc], mutation_token.lemma_, mutation_token_pos)

        if synset is None:
            return curr_solution
        
        valid_synonyms: str = []
        for lemma in synset.lemmas():
            synonym = lemma.name()

            if validate_synonym(mutation_token.text, synonym):
                valid_synonyms.append(synonym)
        
        if not valid_synonyms: 
            return curr_solution
        
        chosen_synonym = random.choice(valid_synonyms)

        if mutation_token.text[0].isupper():
            chosen_synonym = chosen_synonym.capitalize()

        mutaded_prompt = curr_solution.prompt.replace(f'{mutation_token.text}', chosen_synonym)

        neighbor_prompt = mutaded_prompt
        neighbor = Individual(neighbor_prompt)

        return neighbor
    
    def _anneal_individual(self, initial_solution: Individual) -> Individual:
        temp = self._calibrate_temperature(initial_solution)

        logger.info(f'Initial Temperature: {temp:.2f}')

        curr_solution = initial_solution
        self.evaluator.evaluate(curr_solution)

        for i in range(self.max_iterations):
            neighbor = self._generate_neighbor(curr_solution)
            self.evaluator.evaluate(neighbor)

            logger.info(f'Neighbor: {neighbor.prompt} | Fitness: {neighbor.fitness}')

            if neighbor.fitness > curr_solution.fitness:
                curr_solution = neighbor
            else:
                delta_e = curr_solution.fitness - neighbor.fitness
                metropolis_prob = self._metropolis(temp, delta_e)

                if random.random() < metropolis_prob:
                    curr_solution = neighbor
            
            temp = temp * self.cooling_rate

        return curr_solution
    
    def _run(self, initial_population: list[Individual]) -> Individual:
        best_solution = initial_population[0]

        for individual in initial_population:
            logger.info(f'Initial Prompt: {individual.prompt} | Fitness: {individual.fitness}')

            solution = self._anneal_individual(individual)

            logger.info(f'Solution: {solution.prompt} | Fitness: {solution.fitness}')

            if solution.fitness > best_solution.fitness:
                best_solution = solution
        
        return best_solution
