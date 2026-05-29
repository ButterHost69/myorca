## Implement the Orca Paper

### Orca: A Distributed Serving System for Transformer-Based Generative Models: [link](https://www.usenix.org/system/files/osdi22-yu.pdf)

For the current implementation we will be using the model : "openai-community/gpt2-large"

### What is Implemented

- [ ] Iterative Scheduling
- [ ] Selective Batching
- [ ] KV Cacheing

### TODO:

- [ ] Load a Model ; Send a prompt
- [ ] Find a way to add the split and merge batches / tensors
- [ ] Look at mergeable ops
- [ ] Implement the whole split and merge schling
- [ ] Implement KV Cache
