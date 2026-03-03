import json
import torch
from datasets import load_from_disk, Dataset
from unsloth import FastLanguageModel

# ---------------------------------------------------------
# 1. Configuration for Unsloth FunctionGemma (270M)
# ---------------------------------------------------------
max_seq_length = 4096
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

# We use the specific function calling model suggested by Unsloth
model_name = "unsloth/functiongemma-270m-it"

print(f"Loading Unsloth Model: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# ---------------------------------------------------------
# 2. Add LoRA Adapters
# ---------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# ---------------------------------------------------------
# 3. Load & Format Dataset for FunctionGemma
# ---------------------------------------------------------
print("Loading Custom Android JSONL Dataset...")
dataset_path = "./Fine tune DATA/dataset.jsonl"

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

raw_data = load_jsonl(dataset_path)

formatted_data = []
for item in raw_data:
    user_query = item["user_content"]
    tool_name = item["tool_name"]
    try:
        tool_args = json.loads(item["tool_arguments"])
    except:
        tool_args = {}
    
    # Construct the FunctionGemma specific prompt format
    # As per Unsloth documentation for FunctionGemma Mobile Actions
    
    prompt = f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
    
    if tool_name != "none":
        # Format arguments as key:value,key2:value2 (simplified for FunctionGemma)
        args_str = ",".join([f"{k}:{v}" for k, v in tool_args.items()])
        prompt += f"<start_function_call>call:{tool_name}{{{args_str}}}<end_function_call><start_function_response>"
    else:
        # Standard response if no tool is needed
        prompt += f"I cannot help with that based on my tools."
        
    formatted_data.append({"text": prompt})

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(formatted_data)
# Split for training and testing
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]

print(f"Prepared {len(train_dataset)} training samples.")

# ---------------------------------------------------------
# 4. Train the Model using SFTTrainer
# ---------------------------------------------------------
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 20,
        max_steps = 200, # Set to a higher number like 1000 for a real run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("Starting Fine-tuning...")
trainer_stats = trainer.train()

# ---------------------------------------------------------
# 5. Save the trained LoRA Model
# ---------------------------------------------------------
print("Saving LoRA adapters...")
model.save_pretrained("functiongemma-android-lora") # Local saving
tokenizer.save_pretrained("functiongemma-android-lora")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving

print("Done! Model is ready for inference or export to GGUF/Phone.")
