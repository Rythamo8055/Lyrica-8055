"""
FunctionGemma Fine-Tuning for Android Mobile Actions
=====================================================
Uses standard HuggingFace Transformers + PEFT + TRL.
Works reliably on Google Colab Free Tier (T4 GPU).
"""
import os
import sys
import json
import torch
from pathlib import Path
from datasets import Dataset

# ---------------------------------------------------------
# Auto-detect script location and cd into the repo
# This ensures paths work no matter where you run from
# ---------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
os.chdir(SCRIPT_DIR)
print(f"📁 Working directory: {SCRIPT_DIR}")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ---------------------------------------------------------
# 0. Check GPU
# ---------------------------------------------------------
if not torch.cuda.is_available():
    print("=" * 60)
    print("ERROR: No GPU detected!")
    print("Go to Runtime > Change runtime type > Select T4 GPU")
    print("=" * 60)
    raise SystemExit("Please enable GPU and re-run.")
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU Found: {gpu_name} ({gpu_mem:.1f} GB)")

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
model_name = "google/gemma-3-1b-it"  # Small Gemma model for function calling
max_seq_length = 2048
output_dir = "./functiongemma-android-lora"

# 4-bit quantization to fit in free T4 GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ---------------------------------------------------------
# 2. Load Model & Tokenizer
# ---------------------------------------------------------
print(f"\n📥 Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# ---------------------------------------------------------
# 3. Add LoRA Adapters
# ---------------------------------------------------------
print("\n🔧 Adding LoRA adapters...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------
# 4. Load & Format Custom Android Dataset
# ---------------------------------------------------------
print("\n📂 Loading dataset from Fine tune DATA/dataset.jsonl...")
dataset_path = "./Fine tune DATA/dataset.jsonl"

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

raw_data = load_jsonl(dataset_path)
print(f"   Loaded {len(raw_data)} examples")

# Define ALL available Android tools for the system prompt
TOOL_DEFINITIONS = """You are a model that can do function calling with the following functions:
- set_timer(seconds, label): Set a countdown timer
- cancel_timer(): Cancel the current timer
- set_alarm(hour, minute): Set an alarm
- set_brightness(level): Set screen brightness (0.0 to 1.0)
- set_volume(level): Set volume level (0.0 to 1.0)
- toggle_flashlight(state): Turn flashlight on/off
- toggle_wifi(state): Turn WiFi on/off
- toggle_bluetooth(state): Turn Bluetooth on/off
- toggle_airplane_mode(state): Turn airplane mode on/off
- toggle_dnd(state): Turn Do Not Disturb on/off
- toggle_rotation(state): Lock/unlock screen rotation
- make_call(number): Make a phone call
- send_sms(number, message): Send a text message
- send_email(to, subject): Send an email
- open_app(app_name): Open an application
- open_url(url): Open a URL in browser
- speak_text(text): Read text out loud
- stop_speaking(): Stop text-to-speech
- set_speech_rate(rate): Set TTS speaking rate
- vibrate_device(duration_ms): Vibrate the device
- take_screenshot(): Take a screenshot
- show_notification(title, message): Show a notification
- get_battery_info(): Get battery information
- get_device_info(): Get device information
- get_storage_info(): Get storage information
- get_current_time(): Get the current time
- set_clipboard(text): Copy text to clipboard"""

formatted_data = []
for item in raw_data:
    user_query = item["user_content"]
    tool_name = item["tool_name"]
    try:
        tool_args = json.loads(item["tool_arguments"])
    except (json.JSONDecodeError, TypeError):
        tool_args = {}

    # Build the FunctionGemma-style prompt
    prompt = f"<bos><start_of_turn>user\n{TOOL_DEFINITIONS}\n\nUser request: {user_query}<end_of_turn>\n<start_of_turn>model\n"

    if tool_name != "none":
        # Format: call:tool_name{key:value,key2:value2}
        args_parts = []
        for k, v in tool_args.items():
            if isinstance(v, str):
                args_parts.append(f"{k}:{v}")
            else:
                args_parts.append(f"{k}:{v}")
        args_str = ",".join(args_parts)
        prompt += f"<start_function_call>call:{tool_name}{{{args_str}}}<end_function_call><end_of_turn>"
    else:
        prompt += f"I cannot perform that action with my available tools.<end_of_turn>"

    formatted_data.append({"text": prompt})

# Create HuggingFace Dataset and split
dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(f"   Training samples: {len(train_dataset)}")
print(f"   Test samples:     {len(test_dataset)}")

# Show a sample
print("\n📝 Sample formatted prompt:")
print("-" * 50)
print(train_dataset[0]["text"][:500])
print("-" * 50)

# ---------------------------------------------------------
# 5. Training
# ---------------------------------------------------------
print("\n🚀 Starting fine-tuning...")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=20,
    max_steps=200,  # Increase to 500-1000 for better results
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    report_to="none",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=training_args,
)

trainer.train()

# ---------------------------------------------------------
# 6. Save Model
# ---------------------------------------------------------
print(f"\n💾 Saving LoRA adapters to {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ---------------------------------------------------------
# 7. Quick Test
# ---------------------------------------------------------
print("\n🧪 Running a quick inference test...")
test_prompts = [
    "set a timer for 30 seconds called quick",
    "turn on the flashlight",
    "what is the capital of france",
    "call Mom",
    "set brightness to 0.5",
]

model.eval()
for test_query in test_prompts:
    prompt = f"<bos><start_of_turn>user\n{TOOL_DEFINITIONS}\n\nUser request: {test_query}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)
    # Clean up the response
    response = response.split("<end_of_turn>")[0].strip()

    print(f"\n  User: {test_query}")
    print(f"  Model: {response}")

print("\n" + "=" * 60)
print("✅ DONE! Model saved to:", output_dir)
print("=" * 60)
