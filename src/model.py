
from typing import Callable

from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import random
import numpy as np
def set_seed(seed: int = 42):
    # 1. Standard Python and Numpy seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. PyTorch CPU and GPU seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # 3. Force deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class GPT2Large:
    """Mainly for weights loading and for verification"""
    def __init__(self):
        self.model_name = "openai-community/gpt2-large"

        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        ).to("cuda")

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token # This is Temporary
        self.tokenizer.padding_side = "left"  # This is Temporary

    def simple_infer(self, prompts:list[str]):
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt",
            padding=True, # This is Temporary
        ).to(self.model.device)

        # Generate text
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            top_k=50,
            do_sample=False,   # to set temp 0, kinda fixed output
            pad_token_id=self.tokenizer.eos_token_id 
        )

        # Decode output
        generated_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated_text
    
    def print_model(self):
        print(self.model)

    def inspect_conv1d(self):
        import inspect
        from transformers.pytorch_utils import Conv1D
        print(inspect.getsource(Conv1D))
    
    def inspect_mlp(self):
        import inspect
        attn_layer = self.model.transformer.h[0].mlp
        print(inspect.getsource(type(attn_layer)))

    def inspect_attn(self):
        import inspect
        attn_layer = self.model.transformer.h[0].attn
        print(inspect.getsource(type(attn_layer)))
    
    def inspect_transformer(self):
        import inspect
        transformer_layer = self.model.transformer
        print(inspect.getsource(type(transformer_layer)))
    
    def inspect_model(self):
        import inspect
        attn_layer = self.model
        print(inspect.getsource(type(attn_layer)))
    
    def inspect_block(self):
        import inspect
        block_layer = self.model.transformer.h[0]
        print(inspect.getsource(type(block_layer)))

    def register_hooks(self):        
        self.hooks = []
        self.hooks.extend([])
        
        # forward function override (can be done)
        # for block in self.model.transformer.h:
        #     attn = self.model.transformer.h[0].attn
        #     block.attn.forward = self.make_splitattn_forward(attn, self.my_splits)        
            
    def unregister_hooks(self):
        for h in self.hooks:
            h.remove()


DEBUG = False
#  Actual Orca Code Classes
class SplitMerge():
    @staticmethod
    def merge_inputs(inputs:list, shape:list = [1, -1, 1280]):
        """Flattens inputs -> such that output shape is `shape`"""
        if isinstance(inputs, list):
            merged_inputs = torch.cat(inputs, dim=1)
            return merged_inputs.reshape(shape)
    
    @staticmethod
    def split_inputs(inputs, splits:list[int], _dim=1):
        """outputs a tuple containing the split input ; splits are done from indices in splits"""
        return torch.split(inputs, splits, _dim)

class MyOrcaGPT2Block(nn.Module, SplitMerge):
    """ Attention + MLP Class """
    def __init__(self, config, pretrained_block, layer_idx=None):
        super().__init__()
        # Attention Block
        self.ln_1 = pretrained_block.ln_1
        self.c_attn = pretrained_block.attn.c_attn # QKV 
        self.c_proj = pretrained_block.attn.c_proj # Attention Projection
        self.resid_dropout = pretrained_block.attn.resid_dropout

        # Attention Inits
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        # TODO: Find what this even Does ? 
        self.layer_idx = layer_idx
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        # Precompute unified scaling factor (accounts for both head_dim and layer-wise scaling)
        self.scaling = 1.0
        if self.scale_attn_weights:
            self.scaling = self.head_dim**-0.5
        if self.scale_attn_by_inverse_layer_idx:
            self.scaling /= float(self.layer_idx + 1)

        # MLP Block
        self.ln_2 = pretrained_block.ln_2
        self.mlp_c_fc = pretrained_block.mlp.c_fc
        self.mlp_c_proj = pretrained_block.mlp.c_proj
        self.mlp_act = pretrained_block.mlp.act
    
    def compute_attention(self, hidden_states, splits, **kwargs):
        """Basically the forward for the GPTAttention Class"""
        m_query_states, m_key_states, m_value_states = self.c_attn(hidden_states).split(self.split_size, dim=2) # To get 3 embed dim size (split_size = embed_dim)
        if DEBUG : print("    * merged shape: ", m_query_states.shape)
    
        
        query_states = self.split_inputs(m_query_states, splits, _dim=1)
        key_states = self.split_inputs(m_key_states, splits, _dim=1)
        value_states = self.split_inputs(m_value_states, splits, _dim=1)
        
        if DEBUG : print(f"    * q shape: [{len(query_states)}, {query_states[0].shape}]")
        if DEBUG : print(f"    * k shape: [{len(key_states)}, {key_states[0].shape}]")
        if DEBUG : print(f"    * v shape: [{len(value_states)}, {value_states[0].shape}]")
        
        attn_out_list = []
        clean_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ('attention_mask', 'scaling', 'dropout')}
        
        for idx, req_len in enumerate(splits):
            key = key_states[idx]
            value = value_states[idx]
            query = query_states[idx]

            shape_kv = (*key.shape[:-1], -1, self.head_dim)
            key = key.view(shape_kv).transpose(1, 2)
            value = value.view(shape_kv).transpose(1, 2)
            shape_q = (*query.shape[:-1], -1, self.head_dim)
            query = query.view(shape_q).transpose(1, 2)

            using_eager = self.config._attn_implementation == "eager"
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            
            causal_mask = torch.tril(
                torch.ones(req_len, req_len, dtype=torch.bool, device=hidden_states.device)
            ).view(1, 1, req_len, req_len)

            if using_eager and self.reorder_and_upcast_attn:
                raise ValueError("support for eager attention is not done yet :)")
                attn_output, _ = self._upcast_and_reordered_attn(
                    query, key, value, self[idx].squeeze(0).squeeze(0)
                )

            else:
                attn_output, _ = attention_interface(
                    self,
                    query,
                    key,
                    value,
                    causal_mask,
                    dropout= 0.0, # Because not training, so can set directly to 0
                    scaling=self.scaling,
                    **clean_kwargs,
                )
            
            attn_out_list.append(attn_output)

        m_attn_outs = self.merge_inputs(attn_out_list, shape=[1, sum(splits), -1])
        if DEBUG : print("     * merged attn output shape: ", m_attn_outs.shape, "\n")
        m_attn_outs = self.c_proj(m_attn_outs)
        m_attn_outs = self.resid_dropout(m_attn_outs)            
        return m_attn_outs

        

    def forward(self, hidden_states, splits, split_casual_masks, position_embeds):
        if DEBUG : print(f"\n * transfomer layer: {self.layer_idx}")
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output = self.compute_attention(
            hidden_states,
            splits
        )
        
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        
        # MLP pipeline
        ff_hd = self.mlp_c_fc(hidden_states)
        ff_hd = self.mlp_act(ff_hd)
        ff_hd = self.mlp_c_proj(ff_hd)
        
        # residual connection
        hidden_states = residual + ff_hd
        return hidden_states

class MyOrcaGPT2(nn.Module, SplitMerge):
    def __init__(self, pretrained_model):
        """
        - Declare Layers & Load from the GPT2 weights 
        - ignores dropout layers :)
        """
        super().__init__()
        self.config = pretrained_model.config
        self.wte = pretrained_model.transformer.wte
        self.wpe = pretrained_model.transformer.wpe
        
        self.h = nn.ModuleList([
            MyOrcaGPT2Block(config=self.config, pretrained_block=pretrained_model.transformer.h[idx], layer_idx=idx) for idx in range(36) 
        ])

        self.ln_f = pretrained_model.transformer.ln_f
        self.lm_head = pretrained_model.lm_head
    
    @staticmethod
    def logits_to_token(logits, temperature=1.0, top_k=50, top_p=0.9, do_sample=True):
        """
        logits:      [batch, seq_len, vocab_size] or [batch, vocab_size]
        temperature: 0.0 = greedy, <1.0 = sharper, >1.0 = flatter
        top_k:       0 to disable
        top_p:       1.0 to disable (nucleus sampling)
        do_sample:   False = greedy regardless of temperature
        returns:     [batch] token ids
        """
        # grab last token logits if full sequence passed
        if logits.dim() == 3:
            logits = logits[:, -1, :]       # [batch, vocab_size]

        # greedy shortcuts
        if temperature == 0.0 or not do_sample:
            return torch.argmax(logits, dim=-1)

        # temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k, dim=-1)
            min_val = values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_val, float('-inf'))

        # top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # remove tokens once cumulative prob exceeds top_p
            # shift right so the token that pushes over the threshold is kept
            sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))

            # scatter back to original indexing
            logits = torch.zeros_like(logits).scatter_(
                dim=-1,
                index=sorted_indices,
                src=sorted_logits
            )

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)   # [batch]
        return next_token


    @staticmethod
    def logits_to_tokens_split(logits, **kwargs):
        split_logits = logits.squeeze(0)
        return [
            [MyOrcaGPT2.logits_to_token(req_logits, **kwargs).item()]
            for req_logits in split_logits
        ]

    def forward(self, inputs, splits):
        """
        - single forward pass
        - right now, no chacheing ; everything is computed ; support for only sdpa
        - input_ids: contains multiple requests
        """
        split_input_ids = self.split_inputs(inputs["input_ids"], splits)
        if DEBUG : print(" * attn mask: ", inputs["attention_mask"].shape)
        m_input_embeddings = self.wte(inputs["input_ids"]) 
        if DEBUG : print(" * input embeddings: ", m_input_embeddings.shape)
        
        # position embeddings — takes position indices asper split
        s_pos_embeddings = [
            self.wpe(
                torch.arange(
                    req_len,
                    device=inputs["input_ids"].device
                ).unsqueeze(0)
            )
            for req_len in splits
        ]
        
        if DEBUG : print(" * split pos: ", [s_pos_embeddings[idx].shape for idx in range(len(splits))])
        m_pos_embeddings = self.merge_inputs(s_pos_embeddings)
        if DEBUG : print(" * merged pos: ", m_pos_embeddings.shape)
        if DEBUG : print(" * m_input_emd: ", m_input_embeddings[:,0:1, :].shape)


        hidden_states = m_input_embeddings + m_pos_embeddings
        
        start = 0
        split_casual_masks = []
        for idx, end in enumerate(splits):
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=m_input_embeddings[:,start:start+end,:],
                attention_mask=torch.ones(end),
                past_key_values=None,
                position_ids=s_pos_embeddings[idx],
            )
            split_casual_masks.append(causal_mask)
        
        if DEBUG : print(" * causal masks: ", split_casual_masks) # will return if attn type is sdpa --> because the sdpa handles it internally

        # input_ids, splits, split_casual_masks
        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states =hidden_states,
                splits = splits,
                split_casual_masks = causal_mask,
                position_embeds=s_pos_embeddings,
            )
        if DEBUG : print(" * hidden state after all blocks: ", hidden_states.shape)
        
        hidden_states = self.ln_f(hidden_states)
        if DEBUG : print(" * final transformer hs: ", hidden_states.shape)

        end_tokens = []
        start = 0
        for i in splits:
            start += i
            end_tokens.append(start - 1)
        if DEBUG : print(" * final tokens idx: ", end_tokens)
        logits = self.lm_head(hidden_states[:, end_tokens, :])
        if DEBUG : print(" * final logits shape: ", logits.shape)
        return MyOrcaGPT2.logits_to_tokens_split(logits, do_sample=False)
    