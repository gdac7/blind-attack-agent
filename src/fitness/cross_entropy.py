from src.fitness.base import FitnessFunction
from src.models.individual import Individual

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

class CrossEntropyFitness(FitnessFunction):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, target: str = "Sure, here is"):
        self.model = model
        self.tokenizer = tokenizer
        self.target = target

        self.target_ids = self.tokenizer.encode(self.target, add_special_tokens=False)
        self.target_tensor = torch.tensor(self.target_ids, dtype=torch.long, device=self.model.device)

    def evaluate(self, prompt: str) -> float:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        input_ids = prompt_ids + self.target_ids
        input_tensor = torch.tensor([input_ids], device=self.model.device)

        idx_start = len(prompt_ids) - 1
        idx_end = idx_start + len(self.target_ids)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits

        filtered_logits = logits[0, idx_start : idx_end, :]

        loss = F.cross_entropy(filtered_logits, self.target_tensor)
        fitness = -loss.item()

        return fitness
