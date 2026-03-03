import json
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    # 1. Load Configuration
    print("Loading configuration...")
    with open("training_config.json", "r") as f:
        config = json.load(f)

    model_name = config["model_name"]
    output_dir = config["output_dir"]
    
    # 2. Config & Tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # 3. LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Load Processed Dataset
    print("Loading prepared dataset...")
    dataset_train = load_from_disk("./data/train")
    
    # 5. Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        optim=config["optim"],
        save_steps=100,
        logging_steps=10,
        learning_rate=config["learning_rate"],
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=config["max_steps"],
        warmup_ratio=config["warmup_ratio"],
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )
    
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # 6. Train!
    print("Starting training!")
    trainer.train()
    
    # 7. Save
    print(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
