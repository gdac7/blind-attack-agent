from src.optimizers.base import Optimizer
from src.optimizers.fitness.base import FitnessFunction
from src.optimizers.models.particle import Particle
from src.optimizers.models.individual import Individual
from src.optimizers.utils.nlp_constants import TARGET_POS_TAGS, POS_TRANSLATION
from src.optimizers.utils.validate_synonym import validate_synonym
import src.optimizers.nlp.preprocessing as prep

import copy
import random

import spacy
from nltk.wsd import lesk

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
    
    def _update_gbest(self, particle: Particle, curr_gbest: Individual) -> Individual:
        if particle.curr_state.fitness > curr_gbest.fitness:
            return copy.deepcopy(particle.curr_state)
        
        return curr_gbest
    
    def _move_towards_target(self, curr_prompt: str, target_prompt: str, budget: int = 1):
        if budget < 1:
            return curr_prompt

        curr_doc = self.nlp(curr_prompt)
        target_doc = self.nlp(target_prompt)

        curr_core_tokens = prep.valid_tokens(curr_doc)
        target_core_tokens = prep.valid_tokens(target_doc)

        curr_lemma_map = prep.tokens_by_lemma(curr_core_tokens)
        target_lemma_map = prep.tokens_by_lemma(target_core_tokens)

        missing_lemmas = list(set(target_lemma_map.keys()) - set(curr_lemma_map.keys()))

        if not missing_lemmas:
            return curr_prompt

        curr_prompt_tokens = [token.text_with_ws for token in curr_doc]

        random.shuffle(missing_lemmas)

        used_budget = 0
        for lemma, pos in missing_lemmas:
            if used_budget >= budget:
                break

            candidates_idx = [i for i, token in enumerate(curr_doc) if token.pos_ == pos]

            if not candidates_idx:
                continue

            victim_idx = random.choice(candidates_idx)
            victim_punc = curr_doc[victim_idx].whitespace_

            target_word = random.choice(target_lemma_map[(lemma, pos)])
            
            new_word = target_word + victim_punc

            curr_prompt_tokens[victim_idx] = new_word

            used_budget += 1

        return "".join(curr_prompt_tokens)
    
    def _inertial_move(self, prompt: str, budget: int) -> str:
        if budget < 1:
            return prompt
        
        prompt_doc = self.nlp(prompt)
        prompt_tokens = [token.text_with_ws for token in prompt_doc]

        candidates_idx = [i for i, token in enumerate(prompt_doc) if token.pos_ in TARGET_POS_TAGS]

        random.shuffle(candidates_idx)

        used_budget = 0
        for candidate_idx in candidates_idx:
            if used_budget >= budget:
                break

            token = prompt_doc[candidate_idx]
            token_pos = POS_TRANSLATION[token.pos_]

            synset = lesk(
                context_sentence=prompt, 
                ambiguous_word=token.lemma_, 
                pos=token_pos
            )

            if not synset:
                continue

            valid_synonyms = [synonym for synonym in synset.lemma_names() if validate_synonym(token.text, synonym)]

            if not valid_synonyms:
                continue

            chosen_synonym = random.choice(valid_synonyms)
            chosen_synonym = chosen_synonym.replace("_", " ")

            if token.text[0].isupper():
                chosen_synonym = chosen_synonym.capitalize()
            
            token_punc = token.whitespace_

            new_word = chosen_synonym + token_punc

            prompt_tokens[candidate_idx] = new_word

            used_budget += 1

        return "".join(prompt_tokens)

    def _fly_particle(self, particle: Particle, gbest: Individual, iter: int) -> None:
        velocity = self._calc_velocity(particle, gbest, iter)

    def _run(self, initial_population: list[Individual]) -> Individual:
        swarm = self._init_swarm(initial_population)
        gbest = self._find_gbest(swarm)

        for iter in range(self.max_iter):
            for particle in swarm:
                self._evaluate_curr(particle)

                particle.update_pbest()
                gbest = self._update_gbest(particle, gbest)

        pass
