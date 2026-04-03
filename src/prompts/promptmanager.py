from dataclasses import dataclass

@dataclass
class PromptTemplate:
    system_prompt: str
    user_prompt: str
    description: str
    condition: str
    temperature: float 
    max_tokens: int





