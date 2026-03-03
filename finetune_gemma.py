import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    # 1. Configuration
    model_name = "google/gemma-2b-it"  # Can also use 7b
    dataset_name = "glaiveai/glaive-function-calling-v2" # Example function calling dataset
    output_dir = "./gemma-function-calling-lora"
    
    # 2. BitsAndBytes Config for 4-bit quantization (QoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 3. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # 4. LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. Load and process Dataset
    # Here we are just loading a standard function calling dataset. 
    # You might need to preprocess it to match Gemma's chat template.
    dataset = load_dataset(dataset_name, split="train[:5000]") # Subset for faster Colab testing
    
    def format_prompt(sample):
        # This is a highly simplified prompt format for function calling
        # You should adapt this to your exact function calling prompt format
        system_prompt = sample.get('system', '')
        chat = sample.get('chat', '')
        text = f"<bos><start_of_turn>user\n{system_prompt}\n{chat}<end_of_turn>\n<start_of_turn>model\n"
        return {"text": text}
        
    dataset = dataset.map(format_prompt)
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=500, # adjust based on your needs
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )
    
    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # 8. Train!
    trainer.train()
    
    # 9. Save the model
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete! Model saved to", output_dir)

if __name__ == "__main__":
    main()
