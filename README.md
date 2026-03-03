# Android Mobile Actions - FunctionGemma Fine-Tuning

This repository configures an **[Unsloth](https://github.com/unslothai/unsloth) pipeline** to extremely fast fine-tune `unsloth/functiongemma-270m-it` (a 270M parameter Gemma 3 model optimized for mobile deployment). 

We are training this tiny, efficient model on our custom Android system dataset (`dataset.jsonl`) to convert natural language (e.g., `"set a timer for 30 seconds called quick"`) into explicit Android mobile action commands (`call:set_timer{seconds:30,label:quick}`).

## Getting Started in Google Colab

The easiest way to run this is inside a Google Colab notebook.

1. Open [Google Colab](https://colab.research.google.com/) and select an **L4 or T4 GPU**.
2. Run the following to install `unsloth` and clone this repository:

```bash
!pip install unsloth
!git clone https://github.com/Rythamo8055/Lyrica-8055.git
%cd Lyrica-8055
```

3. Ensure your custom dataset is mapped accurately in `./Fine tune DATA/dataset.jsonl`.
4. Run the Unsloth end-to-end training pipeline!

```bash
!python unsloth_functiongemma_train.py
```

## Why Unsloth FunctionGemma?
- **Speed**: Unsloth provides 2x-5x faster training speeds and vastly decreases VRAM usage.
- **Mobile Deployment Ready**: The `270M` parameter model size means we can easily export it to GGUF and run it directly on a Pixel or iPhone device natively with ~50 tokens/s using `llama.cpp` or ExecuTorch.
- **Function Calling**: This specific model heavily favors structure logic out-of-the-box, making it trivial for it to learn mapping user commands to structural calls.
