## Implement the Orca Paper

### Orca: A Distributed Serving System for Transformer-Based Generative Models: [link](https://www.usenix.org/system/files/osdi22-yu.pdf)

For the current implementation we will be using the model : "openai-community/gpt2-large"

Also created a small presentation to explain my friend the paper at surface level: [ppt](./Orca%20Paper.pdf)

Do lmk if I messed up somewhere 👍

### What is Implemented

- [ ] Iterative Scheduling
- [X] Selective Batching
    - [X] Split & Merge layers for the batching
- [ ] Verification with the traditional batching
- [ ] KV Cacheing

### TODO:

- [X] Load a Model ; Send a prompt
- [X] Look at mergeable ops
- [X] Find a way to add the split and merge batches / tensors 
- [X] Implement the whole split and merge schling
  - [x] Implement our own model
  - [X] Load the Weights of the pretrained Model
  - [X] Implement the split and merge inside the forward functions

- [X] Verification with the traditional batching
- [X] Implement a scheduler
    - [ ] Implement the KV count part
- [ ] Concretely display that the whole thing works iteratively
- [ ] Implement the scenarios from paper and compare to maybe out traditional hf model.
- [ ] Implement KV Cache
