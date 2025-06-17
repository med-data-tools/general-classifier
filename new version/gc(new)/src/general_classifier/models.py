import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from openai import OpenAI
import re

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class ModelManager:
    def __init__(self):
        self.model_name = ""
        self.model_type = ""
        self.inference_type = ""
        self.tokenizer = None
        self.llm = None
        self.client = None

    def set_model(self, model_name, model_type="Transformers", api_key="", inference_type="transformers"):
        self.model_name = model_name
        self.model_type = model_type
        self.inference_type = inference_type

        if self.model_type == "Transformers":
            if self.inference_type == "transformers":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                print("Invalid inference Type for Transformers.")
        elif self.model_type == "OpenAI":
            self.inference_type = "cloud"
            if api_key:
                self.client = OpenAI(api_key=api_key)
        elif self.model_type == "DeepInfra":
            self.inference_type = "cloud"
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")

    def get_answer(self, prompt, categories, constrained_output, temperature=0.0, think_step=0):
        if self.inference_type == "cloud":
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=30,
                temperature=temperature,
            )
            generated_answer = completion.choices[0].message.content
            for option in categories:
                escaped_option = re.escape(option)
                if re.search(escaped_option, generated_answer, re.IGNORECASE):
                    return option, "-"
            return "undefined", "-"

        elif self.inference_type == "transformers":
            if think_step > 0:
                # This part needs to be refactored as it uses the old global model
                pass
            
            if constrained_output:
                best_option, _, _, _, best_rel_prob = self._calculate_options_probabilities(prompt, categories)
                return best_option, best_rel_prob
            else:
                # This part needs to be refactored as it uses the old global model
                pass
        return "undefined", "-"

    def _calculate_options_probabilities(self, prompt, options):
        self.llm.eval()
        space_prefix = ' ' if not prompt.endswith(' ') else ''
        
        first_token_groups = {}
        for option in options:
            first_token = self.tokenizer.encode(space_prefix + option, add_special_tokens=False)[0]
            if first_token not in first_token_groups:
                first_token_groups[first_token] = []
            first_token_groups[first_token].append(option)
            
        base_inputs = self.tokenizer(prompt, return_tensors="pt")
        
        option_probabilities = {}
        with torch.no_grad():
            base_outputs = self.llm(**base_inputs)
            base_logits = base_outputs.logits[0, -1, :]
            base_probs = torch.nn.functional.softmax(base_logits, dim=-1)
            
        for first_token_id, group_options in first_token_groups.items():
            if len(group_options) == 1:
                option = group_options[0]
                token_ids = self.tokenizer.encode(space_prefix + option, add_special_tokens=False)
                
                if len(token_ids) == 1:
                    option_probabilities[option] = base_probs[first_token_id].item()
                else:
                    probability, _ = self._calculate_word_probability(prompt, option)
                    option_probabilities[option] = probability
            else:
                for option in group_options:
                    probability, _ = self._calculate_word_probability(prompt, option)
                    option_probabilities[option] = probability
        
        total_probability = sum(option_probabilities.values())
        relative_probabilities = {
            option: (prob / total_probability if total_probability > 0 else 0)
            for option, prob in option_probabilities.items()
        }
        
        best_option = max(option_probabilities.items(), key=lambda x: x[1])
        
        return best_option[0], best_option[1], option_probabilities, relative_probabilities, relative_probabilities[best_option[0]]

    def _calculate_word_probability(self, prompt, target_word):
        self.llm.eval()
        target_word_with_space = ' ' + target_word if not prompt.endswith(' ') else target_word
        target_tokens = self.tokenizer.encode(target_word_with_space, add_special_tokens=False)
        token_probabilities = []
        current_text = prompt

        for token_id in target_tokens:
            inputs = self.tokenizer(current_text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.llm(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token_prob = next_token_probs[token_id].item()
            token_probabilities.append(token_prob)
            current_text = self.tokenizer.decode(
                self.tokenizer.encode(current_text, add_special_tokens=False) + [token_id],
                skip_special_tokens=True
            )
        total_probability = torch.tensor(token_probabilities).prod().item()

        return total_probability, token_probabilities