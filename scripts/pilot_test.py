import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def run_pilot_test():
    print("=== SLM Temperature Pilot Test ===")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model {model_id} on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32, 
        device_map="cpu"
    )
    
    prompt = "If I have 5 apples, and I give 2 to Alice, and Alice gives 1 to Bob, how many apples does Alice have?"
    print("")
    print(f"Prompt: '{prompt}'")
    
    messages = [
        {"role": "system", "content": "You are a helpful mathematical reasoning assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    temperatures = [0.1, 1.0, 1.8]
    
    for temp in temperatures:
        print("")
        print(f"--- Generating with Temperature: {temp} ---")
        start_time = time.time()
        
        do_sample = True if temp > 0.0 else False
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=do_sample,
            temperature=temp,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elapsed = time.time() - start_time
        
        print(f"Time taken: {elapsed:.2f}s")
        print(f"Response: {response.strip()}")

if __name__ == '__main__':
    run_pilot_test()