from abc import ABC, abstractmethod
from src.prompts.prompt_template import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch

class TransformersModel(ABC):
    @property
    @abstractmethod
    def tokenizer(self):
        pass
    
    @property
    @abstractmethod
    def model(self):
        pass

    def generate(self, prompt_template: PromptTemplate) -> str:
        generation_params = {
            "max_new_tokens": prompt_template.max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        if prompt_template.temperature > 0.0:
            generation_params["do_sample"] = True
            generation_params["temperature"] = prompt_template.temperature

        with torch.inference_mode():
            # if LLM has a chat template
            if getattr(self.tokenizer, "chat_template", None):
                try:
                    messages = [
                        {"role": "system", "content": prompt_template.system_prompt},
                        {"role": "user", "content": prompt_template.user_prompt}
                    ] if prompt_template.system_prompt else [{"role": "user", "content": prompt_template.user_prompt}]                    
                    plain_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                # it does have a chat template but without system_prompt
                except Exception as e:
                    messages = [
                        {
                            "role": "user", 
                            "content": f"{prompt_template.system_prompt}\n\n{prompt_template.user_prompt}" if prompt_template.system_prompt else prompt_template.user_prompt
                        }
                    ]
                    plain_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            else:
                plain_text = f"{prompt_template.system_prompt}\n\n{prompt_template.user_prompt}" if prompt_template.system_prompt else prompt_template.user_prompt                
            
            # Condition is simply a text to be inserted at the beginning of the LM response
            plain_text += prompt_template.condition
            inputs = self.tokenizer(plain_text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Get the size of our input
            input_length = inputs['input_ids'].shape[1]
            # Get just the response from LM
            output_ids = self.model.generate(**inputs, **generation_params)
            generated_tokens = output_ids[0][input_length:]
            lm_response = self._wrapper_response(
                self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            )
            return lm_response

    def _wrapper_response(self, lm_response: str) -> str:
        tag = "[END OF THE NEW PROMPT]"
        if tag in lm_response:
            return lm_response.split(tag)[0]
        return lm_response
