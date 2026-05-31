"""
Handles the inference of the Model.
You just interact with it -- handles queueing and scheduling.
"""
from typing import Callable

from transformers import GPT2Tokenizer
from model import GPT2Large, MyOrcaGPT2
from queue import Queue

class Prompt():
    """Stores metadata for prompt"""
    def __init__(self, prompt:str, max_tokens=3):
        self.prompt = prompt
        self.max_tokens = max_tokens

class InferenceEngine():
    def __init__(self):
        self.model_name = "openai-community/gpt2-large"
        self.gpt2 = GPT2Large()
        self.orca = MyOrcaGPT2(self.gpt2.model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.orca_on = False # For the orca inference loop

        self.input_queue = Queue()
        self.max_bs = 3


    def _orca_tokenizer(self, prompts:list[Prompt], device):
        """ Tokenize and split for orca. Split = contains length of each request"""
        inputs = [
            self.tokenizer(prompt.prompt, return_tensors="pt") for prompt in prompts
        ]

        input_ids = [
            inputs[idx]["input_ids"] for idx in range(len(inputs))
        ]

        attn_mask = [
            inputs[idx]["attention_mask"] for idx in range(len(inputs))
        ]

        splits = [len(i[0]) for i in inputs]
        merged_input_ids = MyOrcaGPT2.merge_inputs(input_ids, [1, -1]).to(device)
        merged_attn_mask = MyOrcaGPT2.merge_inputs(attn_mask, [1, -1]).to(device)
        return {
            "input_ids": merged_input_ids,
            "attention_mask": merged_attn_mask
        }, splits  
    
    def _orca_iteration(self, inputs, splits):
        """Takes in a batch of inputs and performs a single forward pass / iteration -> output: generated tokens"""
        outputs = self.orca.forward(
            inputs,
            splits
        )
        
        return self.tokenizer.decode(outputs, skip_special_tokens=True)
    
    def process_prompt(self, prompt:str|Prompt|list[str|Prompt]):
        if isinstance(prompt, list):
            for p in prompt:
                if isinstance(p, str):
                    self.input_queue.put(Prompt(p))
                else:
                    self.input_queue.put(p)  
        else:
            if isinstance(prompt, str):
                self.input_queue.put(Prompt(prompt))
            else:
                self.input_queue.put(prompt)

    
    def stop_orca(self):
        self.orca_off = False

    def _select_batch(self):
        """
        - Right Now acts as a simple first come first serve select
        - TODO: Will implement the rsrv later
        - BTW Blocks until the batch sized satisfied
        """
        batch = []
        while True:
            if self.input_queue.empty(): # if none left compute the remaining in the batch
                break
            
            req = self.input_queue.get() # Blocks till fetched value
            batch.append(req)
            if len(batch) == self.max_bs:
                break
        tokenized = self._orca_tokenizer(batch, "cuda")
        return batch, tokenized[0], tokenized[1]


    def start_orca(self, post_iter_hook:Callable=None,yield_mode=False):
        """
        - Loops until stop_orca() not called.
        - Processes inputs inside the input queue
        - Selects the relevant batch on the basis of first come first serve
        - post_iter_hook : is a callable function triggered after every iteration
        """
    
        self.orca_on = True
        while self.orca_on:
            prompts, inputs, splits = self._select_batch() # Blocking
            outputs = self._orca_iteration(inputs, splits)

            new = []
            outs = []
            input_ids = MyOrcaGPT2.split_inputs(inputs["input_ids"], splits)
            for p, out, inp in zip(prompts, outputs, input_ids):
                # TODO: See outputs that have generated - <eos> -> return / yield those
                if len(inp[0]) + 1 >= p.max_tokens:
                    outs.append(p.prompt + out)
                else:
                    new.append(p.prompt + out)

            self.process_prompt(new)
            yield outs
            