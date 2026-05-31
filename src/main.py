from model import GPT2Large
from model import set_seed
from model import MyOrcaGPT2
import utils

from transformers import GPT2Tokenizer

def orca_tokenizer(tokenizer:GPT2Tokenizer, prompts:list[str], device):
    """ Tokenize and split for orca. Split -  contains length of each request"""
    inputs = [
        tokenizer(prompt, return_tensors="pt") for prompt in prompts
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


def orca_inference(model, tokenizer:GPT2Tokenizer, prompts:list[str]):
    inputs, splits = orca_tokenizer(tokenizer, prompts, "cuda")
    utils.print_inputs_decode(inputs, tokenizer)
    
    outputs = model.forward(
        inputs,
        splits
    )
    # Decode output
    generated_text = tokenizer.decode(outputs, skip_special_tokens=True)
    print("Generated text: ", generated_text)
    return [p+g for p, g in zip(prompts, generated_text)]


def my_orca():
    set_seed(42)
    print("Hello from myorca! - def myorca()")
    prompts = ["Capital of India is", "the shortest King", "The Cat Sat on the"]
    small_prompts = ["damn", "i am", "you are so"]

    print("GPT2 Model: ")
    gpt2_large = GPT2Large()
    print("Promts: ", small_prompts)
    print("========= Normal Call =========")
    print(gpt2_large.simple_infer(prompts=small_prompts))
    print("========= =========== =========\n\n\n")

    set_seed(42)
    print("========= My Orca Call =========")
    model_name = "openai-community/gpt2-large"
    my_orca = MyOrcaGPT2(gpt2_large.model)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    print("Final Output: ", orca_inference(my_orca, tokenizer, small_prompts))
    


if __name__ == "__main__":
    my_orca()
