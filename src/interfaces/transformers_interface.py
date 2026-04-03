from abc import ABC, abstractmethod
from 
class TransformersModel(ABC):
    @abstractmethod
    def generate(self, ):
        pass

    def _wrapper_response(self, lm_response: str) -> str:
        tag = "[END OF THE NEW PROMPT]"
        if tag in lm_response:
            return lm_response.split(tag)[0]
        return lm_response
