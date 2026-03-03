import json
from datasets import load_dataset, Dataset

def validate_dataset(dataset):
    print("Validating dataset...")
    missing_fields = 0
    for item in dataset:
        if 'system' not in item or 'chat' not in item:
            missing_fields += 1
            
    if missing_fields > 0:
        print(f"Warning: {missing_fields} items missing 'system' or 'chat' fields.")
    else:
        print("Validation passed. All items have required fields.")
        
def format_prompt(sample):
    # Gemma prompt format
    system_prompt = sample.get('system', '')
    chat = sample.get('chat', '')
    
    # We construct a Gemma-compatible instruction format
    text = f"<bos><start_of_turn>user\n{system_prompt}\n{chat}<end_of_turn>\n<start_of_turn>model\n"
    return {"text": text}

def main():
    dataset_name = "glaiveai/glaive-function-calling-v2"
    print(f"Loading {dataset_name}...")
    
    # Load separate train and test splits for evaluation later
    dataset_train = load_dataset(dataset_name, split="train[:4000]")
    dataset_test = load_dataset(dataset_name, split="train[4000:4500]") 
    
    validate_dataset(dataset_train)
    
    dataset_train = dataset_train.map(format_prompt)
    dataset_test = dataset_test.map(format_prompt)
    
    print("Saving processed datasets...")
    dataset_train.save_to_disk("./data/train")
    dataset_test.save_to_disk("./data/test")
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
