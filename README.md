## Implement the Orca Paper

### Orca: A Distributed Serving System for Transformer-Based Generative Models: [link](https://www.usenix.org/system/files/osdi22-yu.pdf)

For the current implementation we will be using the model : "openai-community/gpt2-large"

Also created a small presentation to explain my friend the paper at surface level: [ppt](./Orca%20Paper.pdf)

Do lmk if I messed up somewhere 👍

### What is Implemented

- [ ] Iterative Scheduling -> implemented partially ; not working rn
- [x] Selective Batching
  - [x] Split & Merge layers for the batching
- [ ] Verification with the traditional batching
- [X] KV Cacheing

### TODO:

- [x] Load a Model ; Send a prompt
- [x] Look at mergeable ops
- [x] Find a way to add the split and merge batches / tensors
- [x] Implement the whole split and merge schling
  - [x] Implement our own model
  - [x] Load the Weights of the pretrained Model
  - [x] Implement the split and merge inside the forward functions

- [x] Verification with the traditional batching
- [x] Implement a scheduler
  - [ ] Implement the KV count part
- [ ] Concretely display that the whole thing works iteratively
- [ ] Implement the scenarios from paper and compare to maybe out traditional hf model.
- [X] Implement KV Cache

#### What shall i do ?

<!-- - Look into how kv is managed  -->
<!-- - Create a plan to implement -->
<!-- - Modify existing code to accomodate it -->

- modify the \_select_batch(), to now be mindful of it

- let ig gpt generate the 2 load scenarios -- remmeber to use the poisson generation or something
