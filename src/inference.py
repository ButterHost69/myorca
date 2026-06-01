"""
Handles the inference of the Model.
You just interact with it -- handles queueing and scheduling.
"""
from typing import Callable

import torch
from transformers import GPT2Tokenizer
from model import GPT2Large, MyOrcaGPT2
from queue import Queue

class Prompt():
    """Stores metadata for prompt"""
    def __init__(self, prompt:str, max_new_tokens=3, n_attn_blocks=36):
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.initial_tokens = 0 # Set during tokenization
        self.tokens_processed = 0

        self.prev_kv = [None] * n_attn_blocks
        self.processing_input = prompt # This is what will be processed, will be switched with the next token, when in decoding phase
        

class InferenceEngine():
    def __init__(self):
        self.model_name = "openai-community/gpt2-large"
        self.gpt2 = GPT2Large()
        self.orca = MyOrcaGPT2(self.gpt2.model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.orca_on = False # For the orca inference loop

        self.input_queue = Queue()
        self.max_bs = 3
        self.n_rsrv = 0 # Current KV slots reserved
        self.kv_slots = 10 # Total kv count across all requests ; for each req -> req.n_tokens + max_new_tokens


    def _orca_tokenizer(self, prompts:list[Prompt], device):
        """ Tokenize and split for orca. Split = contains length of each request"""
        inputs = [
            self.tokenizer(prompt.processing_input, return_tensors="pt") for prompt in prompts
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
    
    def _orca_iteration(self, inputs, splits, offset, past_kvs):
        """Takes in a batch of inputs and performs a single forward pass / iteration -> output: generated tokens"""
        outputs, kv = self.orca.forward(
            inputs,
            splits,
            offset,
            past_kvs
        )
        
        # kvs -> tuple[per_req_kvs]
        return kv, self.tokenizer.decode(outputs, skip_special_tokens=True)
    
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
            if req.tokens_processed == 0:
                new_n_rsrv = req.initial_tokens + req.max_new_tokens
                if new_n_rsrv > self.kv_slots:
                    break    
                self.n_rsrv = new_n_rsrv
             
            batch.append(req)
            if len(batch) == self.max_bs:
                break

        (tokenized, splits) = self._orca_tokenizer(batch, "cuda")
        for p,s in zip(batch, splits):
            p.initial_tokens = s
        return batch, tokenized, splits


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
            print(" # Total Tokens Processing: ", sum(splits))
            kvs, outputs = self._orca_iteration(
                inputs, 
                splits,
                [p.tokens_processed for p in prompts],
                [p.prev_kv for p in prompts]
            )

            # Total KV memory across all requests and layers
            total_tokens = sum(kvs[req][0][0].shape[1] for req in range(len(kvs)))
            k0, v0 = kvs[0][0]
            kv_width = k0.shape[-1] + v0.shape[-1]  # concat k and v on last dim
            print(f" # KV cache shape: (1, {total_tokens}, {len(self.orca.h)}, {kv_width})")

            new = []
            outs = []
            for p, out, kv in zip(prompts, outputs, kvs):
                # TODO: See outputs that have generated - <eos> -> return / yield those
                p.prev_kv = kv
                p.tokens_processed += 1
                if p.tokens_processed >= p.max_new_tokens:
                    outs.append(p.prompt + out)
                    self.n_rsrv -= p.initial_tokens + p.max_new_tokens
                else:
                    # Now only process the new tokens
                    p.processing_input = out
                    p.prompt += out
                    new.append(p)

            self.process_prompt(new)
            yield outs
            