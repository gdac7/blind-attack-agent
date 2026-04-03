from abc import ABC, abstractmethod
from prompts.prompt_template import PromptTemplate
class TransformersModel(ABC):
    @abstractmethod
    def generate(self, prompt_template: PromptTemplate) -> str:
        pass

    def _wrapper_response(self, lm_response: str) -> str:
        tag = "[END OF THE NEW PROMPT]"
        if tag in lm_response:
            return lm_response.split(tag)[0]
        return lm_response
