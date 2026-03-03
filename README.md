# Android Mobile Actions - FunctionGemma Fine-Tuning

Fine-tune a small Gemma model on your custom Android system dataset (`dataset.jsonl`) to convert natural language into Android mobile action commands.

## Google Colab Setup (Free T4 GPU)

### Step 1: Enable GPU
Go to **Runtime → Change runtime type → T4 GPU → Save**

### Step 2: Install & Clone
```bash
!pip install transformers peft trl accelerate bitsandbytes datasets
!git clone https://github.com/Rythamo8055/Lyrica-8055.git
%cd Lyrica-8055
```

### Step 3: Train
```bash
!python unsloth_functiongemma_train.py
```

The script will automatically:
- Check GPU is connected
- Load & format your `dataset.jsonl`
- Train with LoRA (4-bit quantized)
- Evaluate on a held-out test set
- Run inference tests when done
