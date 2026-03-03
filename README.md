# Gemma Function Calling Fine-Tuning

This repository contains the boilerplate code needed to fine-tune a Google Gemma model (e.g., `gemma-2b-it` or `gemma-7b-it`) for function-calling capabilities using Google Colab.

## Getting Started in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Make sure to select a GPU runtime (T4 GPU is freely available and sufficient for `gemma-2b-it` with LoRA).
4. Clone this repository in the first cell of your Colab notebook:

```bash
!git clone https://github.com/Rythamo8055/Lyrica-8055.git
%cd Lyrica-8055
```

5. Install the required dependencies:

```bash
!pip install -r requirements.txt
```

6. Authenticate with Hugging Face (Required to download Gemma models):

```python
from huggingface_hub import notebook_login
notebook_login()
```
*Note: You need to accept the Gemma license on Hugging Face before downloading the model.*

7. Run the fine-tuning script!

```bash
!python finetune_gemma.py
```

## Modifying the script
- Edit `finetune_gemma.py` to change the `dataset_name`, adjust `max_steps`, or customize the prompt formatting (`format_prompt` function) to match your specific function-calling data structure.
- The script uses `BitsAndBytesConfig` for 4-bit quantization, allowing it to fit into the standard Colab T4 GPU VRAM.
