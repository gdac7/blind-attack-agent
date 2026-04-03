from dataclasses import dataclass

@dataclass
class ShortTermMemory:
    malicious_request: str
    jailbreak_prompt: str
    score: float

    def __post_init__(self):
        if not(0 <= self.score <= 10):
            raise ValueError(f"score must be between 0 and 10, got {self.score}")