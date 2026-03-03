import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import time

def evaluate():
    with open("training_config.json", "r") as f:
        config = json.load(f)

    base_model_name = config["model_name"]
    lora_model_path = config["output_dir"]
    
    print("Loading test dataset...")
    test_data = load_from_disk("./data/test")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    
    print("Starting evaluation...")
    sample = test_data[0]['text']
    prompt = sample.split("<start_of_turn>model\n")[0] + "<start_of_turn>model\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
    end_time = time.time()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- TEST PROMPT ---")
    print(prompt)
    print("\n--- MODEL RESPONSE ---")
    print(response[len(prompt):])
    print(f"\nInference Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    evaluate()
