from inference import InferenceEngine
from inference import Prompt
from model import set_seed

prompts = ["Capital of India is", "the shortest King", "The Cat Sat on the"]
small_prompts = ["hello", "i", "you"]
small_prompts = ["hello, the sun", "has risen", "but it seem you havent"]
def my_orca():
    set_seed(42)
    print("Hello from myorca! - def myorca()")
    engine = InferenceEngine()
    print("max_new_tokens: 3")
    print("input prompts: ", small_prompts)
    print()
    print("Normal GPT2 Response: ", engine.gpt2.simple_infer(small_prompts))
    print("\nOrca Responses")
    engine.process_prompt(small_prompts)
    for output in engine.start_orca():
        print("out: ", output)
        if engine.input_queue.empty():
            engine.stop_orca()
            break
    
    


if __name__ == "__main__":
    my_orca()
