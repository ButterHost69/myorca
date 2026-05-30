"""
Random functions i dont want to remove 
"""

def print_inputs_decode(inputs, tokenizer):
    decode_io = {
        inputs["input_ids"][0,idx]:tokenizer.decode(inputs["input_ids"][0,idx], skip_special_tokens=True) for idx in range(len(inputs["input_ids"][0]))
    } 
    print(f" * Decoded Input: \n\t{'\n\t'.join([f"{i}:{decode_io[i]}" for i in decode_io.keys()])}")