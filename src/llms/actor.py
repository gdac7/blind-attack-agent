from src.interfaces.transformers_interface import TransformersModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch

class ActorTM(TransformersModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa"
        )
        
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def model(self):
        return self._model
    

