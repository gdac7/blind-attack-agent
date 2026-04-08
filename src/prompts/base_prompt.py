import textwrap
from abc import ABC, abstractmethod
from src.prompts.prompt_template import PromptTemplate

class BasePrompt(ABC):
    def __init__(self, temperature: float, max_tokens: int):
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def get_prompt(self, **kwargs) -> PromptTemplate:
        pass