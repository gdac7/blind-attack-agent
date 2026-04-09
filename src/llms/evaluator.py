from src.interfaces.transformers_interface import TransformersModel
from src.prompts.evaluator_prompt import EvaluatorPrompt

class EvaluatorTM(TransformersModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.prompt_manager = EvaluatorPrompt() 
    
    def evaluate(self, jailbreak_prompt: str, original_malicious_request: str) -> str:
        prompt_template = self.prompt_manager.get_prompt(
            jailbreak_prompt=jailbreak_prompt,
            malicious_request=original_malicious_request,
        )
        eval_json = self.generate(prompt_template=prompt_template)
        return eval_json
