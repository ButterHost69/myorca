## Implement the Orca Paper

### Orca: A Distributed Serving System for Transformer-Based Generative Models: [link](https://www.usenix.org/system/files/osdi22-yu.pdf)

For the current implementation we will be using the model : "openai-community/gpt2-large"

### What is Implemented

- [ ] Iterative Scheduling
- [ ] Selective Batching
    - [ ] Split & Merge layers for the batching
- [ ] KV Cacheing

### TODO:

- [X] Load a Model ; Send a prompt
- [X] Look at mergeable ops
- [X] Find a way to add the split and merge batches / tensors 
- [ ] Implement the whole split and merge schling
- [ ] Implement a scheduler
- [ ] Concretely display that the whole thing works iteratively
- [ ] Implement KV Cache


### GPT 2 Architecture:
```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 1280)
    (wpe): Embedding(1024, 1280)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-35): 36 x GPT2Block(
        (ln_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=3840, nx=1280) -> Same as Linear Layers
          (c_proj): Conv1D(nf=1280, nx=1280) -> Same as Linear Layers
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=5120, nx=1280)      -> Same as Linear Layers
          (c_proj): Conv1D(nf=1280, nx=5120)    -> Same as Linear Layers
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1280, out_features=50257, bias=False)
)
```
