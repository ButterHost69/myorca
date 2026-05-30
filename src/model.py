
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
    
    # 4. Optional: Force newer PyTorch versions to use deterministic operations
    # torch.use_deterministic_algorithms(True)

set_seed(42)


class Model:
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
    
    def inspect_attn(self):
        import inspect
        attn_layer = self.model.transformer.h[0].attn
        print(inspect.getsource(type(attn_layer)))

    def calculate_split(self,prompts:list[str]):
        tokens = self.tokenize_seperatly(prompts)
        # List of end idx for each token
        splits = [len(i[0]) for i in tokens]
        return splits
    
    def register_splits(self, splits:list[int]):
        self.my_splits = splits

    def register_hooks(self):
        def merge_hook(name:str):
            def hook(module, inputs): 
                print(name)
                print(" * merge hook called at:", module._get_name())
                # input[0], becacuse input is a tuple with one element ???
                embed_inputs = inputs[0]
                print("     * raw inputs: ", embed_inputs.shape) 

                # TODO: Doesnt each operation eat new memory ? fix that
                merged_outs = embed_inputs.clone() 
                merged_outs = embed_inputs.reshape(-1, 1280) 
                print("     * merged: ", merged_outs.shape) # -1, 1280 -> any size, 1280
                merged_outs = merged_outs.unsqueeze(0) # Have to match output dim
                print("     * final out", merged_outs.shape, "\n")
                inputs = list(inputs)
                inputs[0] = merged_outs
                return tuple(inputs)


            return hook
        
        self.hooks = [
            self.model.transformer.h[0].register_forward_pre_hook(merge_hook(f"merged before h"))
        ]
        
        for block in self.model.transformer.h:
            attn = self.model.transformer.h[0].attn
            block.attn.forward = self.make_splitattn_forward(attn, self.my_splits)        
            
    def unregister_hooks(self):
        for h in self.hooks:
            h.remove()
        print("\nunregistered all hooks")


    def tokenize_seperatly(self, prompts:list[str]):
       return [
           self.tokenizer(
                prompt, 
                return_tensors="pt",
            ).to(self.model.device) for prompt in prompts
        ] 

    @staticmethod
    def make_splitattn_forward(attn_module, my_splits):        
        def _splitattn_forward(*args, **kwargs):
            print("my attn")
            hidden_states = args[0] if args else kwargs.get('hidden_states', None)
            assert hidden_states is not None, "hidden states cannot be None"

            # Merged Attention
            attention_mask = args[2] if len(args) > 2 else kwargs.get('attention_mask')
            assert attention_mask is not None, "attention mask cannot be None"
            
            # TODO:There are problems with it as torch.split generates tuples, make it into tensor
            if attention_mask.ndim == 2:
                print(" * splitting attention mask: ", attention_mask.shape)
                attention_mask = torch.split(attention_mask, my_splits, dim=0)
                print(" * split attention mask: ", attention_mask.shape)

            # using_eager = attn_module.config._attn_implementation == "eager"
            print(" * attn type: ", attn_module.config._attn_implementation)

            # Merged query, key, values
            m_query_states, m_key_states, m_value_states = attn_module.c_attn(hidden_states).split(attn_module.split_size, dim=2)
            print(" * merged shape: ", m_query_states.shape)
        
            # Split -> tuple [1, num tokens, embed_dim]
            query_states = torch.split(m_query_states, my_splits, dim=1)
            key_states = torch.split(m_key_states, my_splits, dim=1)
            value_states = torch.split(m_value_states, my_splits, dim=1)

            # attention_mask = torch.split(m_attention_mask, my_splits, dim=0)
            print(f" * q shape: [{len(query_states)}, {query_states[0].shape}]")
            print(f" * k shape: [{len(key_states)}, {key_states[0].shape}]")
            print(f" * v shape: [{len(value_states)}, {value_states[0].shape}]")
            print(f" * attention mask: [{len(attention_mask)}, {attention_mask[0].shape}]")

            attn_out_list = []
            attn_weight_list = []

            clean_kwargs = {k: v for k, v in kwargs.items() 
                            if k not in ('attention_mask', 'scaling', 'dropout')}

            for idx in range(len(query_states)):
                key = key_states[idx]
                value = value_states[idx]
                query = query_states[idx]
                
                shape_kv = (*key.shape[:-1], -1, attn_module.head_dim)
                key = key.view(shape_kv).transpose(1, 2)
                value = value.view(shape_kv).transpose(1, 2)

                shape_q = (*query.shape[:-1], -1, attn_module.head_dim)
                query = query.view(shape_q).transpose(1, 2)


                using_eager = attn_module.config._attn_implementation == "eager"
                attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                    attn_module.config._attn_implementation, eager_attention_forward
                )

                if using_eager and attn_module.reorder_and_upcast_attn:
                    attn_output, attn_weights = attn_module._upcast_and_reordered_attn(
                        query, key, value, attention_mask[idx].squeeze(0).squeeze(0)
                    )
                    
                else:
                    attn_output, attn_weights = attention_interface(
                        attn_module,
                        query,
                        key,
                        value,
                        attention_mask[idx].squeeze(1).squeeze(1),
                        dropout= 0.0, # Because not training
                        scaling=attn_module.scaling,
                        **clean_kwargs,
                    )
                
                
                # Merge here
                attn_out_list.append(attn_output)
                attn_weight_list.append(attn_weights)
            
            m_attn_outs    = torch.cat(attn_out_list, dim=0)
            m_attn_weights = torch.cat(attn_weight_list, dim=0) if attn_weight_list[0] is not None else None
            
            m_attn_outs = m_attn_outs.reshape(*m_attn_outs.shape[:-2], -1).contiguous()
            m_attn_outs = m_attn_outs.reshape(1, -1, 1280)
            print(" * merged attn output shape: ", m_attn_outs.shape, "\n")
            m_attn_outs = attn_module.c_proj(m_attn_outs)
            m_attn_outs = attn_module.resid_dropout(m_attn_outs)            
            return m_attn_outs, m_attn_weights
        return _splitattn_forward