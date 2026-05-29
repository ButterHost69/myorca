from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Model:
    def __init__(self):
        self.model_name = "openai-community/gpt2-large"
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch

        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        ).to("cuda")

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

    def simple_infer(self, prompt:str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate text
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def print_model(self):
        print(self.model)