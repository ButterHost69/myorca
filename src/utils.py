"""
Random functions i dont want to remove 
"""

def print_inputs_decode(inputs, tokenizer):
    decode_io = {
        inputs["input_ids"][0,idx]:tokenizer.decode(inputs["input_ids"][0,idx], skip_special_tokens=True) for idx in range(len(inputs["input_ids"][0]))
    } 
    print(f" * Decoded Input: \n\t{'\n\t'.join([f"{i}:{decode_io[i]}" for i in decode_io.keys()])}")



# Legacy Hooks, not used in the main code, but can be used for debugging and testing.
def inspect_hook(name:str):
    def hook(module, inputs, outputs):
        print(name)
        print("inpecting: ", module._get_name())
        if hasattr(inputs, "shape"):
            print(f" * raw inputs: {inputs.shape}")
        else:    
            print(f" * raw inputs: [{len(inputs)}, {inputs[0].shape}]")
        
        if hasattr(outputs, "shape"):
            print(f" * raw outputs: {outputs.shape}\n")
        else:
            print(f" * raw outputs: [{len(outputs)}, {outputs[0].shape}]\n")
    return hook


def wpe_pre_split_hook(name:str):
    def hook(module, inputs):
        print(name)
        print(f" * raw inputs: [{len(inputs)}, {inputs[0].shape}]")
        print(f" * split shape: ")
        return inputs
    return hook
    
def wpe_post_split_hook(name:str):
    def hook(module, inputs):
        print(name)
        print(f" * raw inputs: [{len(inputs)}, {inputs[0].shape}]")
    return hook
        
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