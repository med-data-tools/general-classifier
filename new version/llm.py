import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
from dotenv import load_dotenv

def load_model():
    load_dotenv()  # Load environment variables from .env
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    print("Loading model and tokenizer...")
    model_name = "openai-community/gpt2"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    
    # Load model with 8-bit quantization to reduce memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        # load_in_8bit=True,      # Temporarily remove this
        token=hf_token,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=100):
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generation_time = time.time() - start_time
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, generation_time

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    print("\nModel loaded successfully! You can now start chatting.")
    print("Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        response, gen_time = generate_response(model, tokenizer, user_input)
        print(f"\nAssistant: {response}")
        print(f"\nGeneration time: {gen_time:.2f} seconds")

if __name__ == "__main__":
    main() 