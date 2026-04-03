from interfaces.transformers_interface import TransformersModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch

class ActorTransformersModel(TransformersModel):
    def __init__(self, model_url: str, device: str = "auto"):
        self.model_name = model_url
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_url, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            torch_dtype="auto",
        )

    def generate(self, user_prompt: str, system_prompt: str, temperature: float = 0.7, 
                 max_tokens: str = 4096, condition: str = "") -> str:
        generation_params = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        if temperature > 0.0:
            generation_params["do_sample"] = True
            generation_params["temperature"] = temperature

        with torch.inference_mode():
            # if LM has a chat template
            if getattr(self.tokenizer, "chat_template", None):
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ] if system_prompt else [{"role": "user", "content": user_prompt}]                    
                    plain_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                # it does have a chat template but without system_prompt
                except:
                    messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt}]
                    plain_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            else:
                plain_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt                
            
            # Condition is simply a text to be inserted at the beginning of the LM response
            plain_text += condition
            inputs = self.tokenizer(plain_text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Get the size of our input
            input_length = inputs['input_ids'].shape[1]
            # Get just the response from LM
            output_ids = self.model.generate(**inputs, **generation_params)
            generated_tokens = output_ids[0][input_length:]
            lm_response = self._wrapper_response(
                self.tokenize.decode(generated_tokens, skip_special_tokens=True).strip()
            )
            return lm_response
        


