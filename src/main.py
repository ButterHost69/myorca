from model import Model
from model import set_seed
from model import MyOrcaGPT2
import utils

from transformers import GPT2Tokenizer


def main():
    set_seed(42)
    print("Hello from myorca!")
    prompts = ["Capital of India is", "the shortest King", "The Cat Sat on the"]
    small_prompts = ["hello", "world is", "how are you"]

    print("GPT2 Model: ")    
    my_model = Model()
    my_model.print_model()
    print("\n")
    # print("GPT2 Attention Code:")
    # my_model.inspect_attn()
    print("GPT2 LM Head Code:")
    my_model.inspect_model()
    print()

    print("========= Normal Call =========")
    print(my_model.simple_infer(prompts=small_prompts))
    print(my_model.simple_infer(prompts=small_prompts))
    print("========= =========== =========\n\n\n")


    print("========= ORCA CALL ===========")
    # tokens = my_model.tokenize_seperatly(small_prompts)
    # for p, t in zip(small_prompts, tokens):
    #     print(f" * {p}: {len(t[0])}")
    # print("\n")
    # # TODO: Merge the 2 split functions into 1
    # splits = my_model.calculate_split(small_prompts)
    # my_model.register_splits([3, 3, 3])
    # print("Splits: ", splits)
    # print("\n\n")
    # my_model.register_hooks()
    # print(my_model.simple_infer(prompts=small_prompts))
    # my_model.unregister_hooks()
    print(my_model.orca_infer(prompts=small_prompts))
    print("========= =========== =========\n\n\n")

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
    return generated_text


def my_orca():
    set_seed(42)
    print("Hello from myorca! - def myorca()")
    prompts = ["Capital of India is", "the shortest King", "The Cat Sat on the"]
    small_prompts = ["hello", "world is", "how are you"]

    print("GPT2 Model: ")
    my_model = Model()

    print("GPT2 Transfomer Code:")
    my_model.inspect_block()
    print("\n\n")

    print("========= Normal Call =========")
    print(my_model.simple_infer(prompts=small_prompts))
    print("========= =========== =========\n\n\n")

    print("========= My Orca Call =========")
    
    model_name = "openai-community/gpt2-large"
    my_orca = MyOrcaGPT2(my_model.model)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    print(orca_inference(my_orca, tokenizer, small_prompts))
    


if __name__ == "__main__":
    my_orca()
