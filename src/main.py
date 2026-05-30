from model import Model
from model import set_seed

def main():
    set_seed(42)
    print("Hello from myorca!")
    prompts = ["Capital of India is", "the shortest King", "The Cat Sat on the"]
    small_prompts = ["hello", "world is", "how are you"]

    print("GPT2 Model: ")    
    my_model = Model()
    my_model.print_model()
    print("\n")
    print("GPT2 Attention Code:")
    my_model.inspect_attn()
    print()

    print("========= Normal Call =========")
    print(my_model.simple_infer(prompts=small_prompts))
    print("========= =========== =========\n\n\n")


    print("========= ORCA CALL ===========")
    tokens = my_model.tokenize_seperatly(small_prompts)
    for p, t in zip(small_prompts, tokens):
        print(f" * {p}: {len(t[0])}")
    print("\n")
    # TODO: Merge the 2 split functions into 1
    splits = my_model.calculate_split(small_prompts)
    my_model.register_splits([3, 3, 3])
    print("Splits: ", splits)
    print("\n\n")
    my_model.register_hooks()
    print(my_model.simple_infer(prompts=small_prompts))
    my_model.unregister_hooks()
    print("========= =========== =========\n\n\n")



if __name__ == "__main__":
    main()
