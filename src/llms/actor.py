from src.interfaces.transformers_interface import TransformersModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch
from src.memory.shot_term_memory import ShortTermMemory
from src.memory.long_term_memory import LongTermMemory
from src.prompts.actor_prompt import ActorPrompt
import uuid
import json
from pathlib import Path

class ActorTM(TransformersModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        self.short_term_memory: ShortTermMemory = None
        self.long_term_memory: list[LongTermMemory] = []
        self.attack_history: list[ShortTermMemory] = []
        self.attack_history_path = "data/actor_history/attacks.json"
        self.prompt_manager = ActorPrompt
    
    def attack(self, malicious_request: str) -> str:
        prompt_template = self.prompt_manager.get_prompt(malicious_request)
        jailbreak_prompt = self.generate(prompt_template=prompt_template)
        score = 0 # Change for evaluator score
        self.short_term_memory = ShortTermMemory(
            malicious_request=malicious_request,
            jailbreak_prompt=jailbreak_prompt,
            score=score
        )
        self.attack_history.append(self.short_term_memory)
        return jailbreak_prompt

    def save_attack_history(self):
        Path(self.attack_history_path).mkdir(parents=True, exist_ok=True)
        with open(self.attack_history_path, "w", encoding="utf-8") as f:
            json.dump([attack.__dict__ for attack in self.attack_history], indent=4)

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def model(self):
        return self._model
    

