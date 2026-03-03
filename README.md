# Gemma Function Calling Fine-Tuning

This repository follows the **Fine-Tuning Expert** guidelines to train a Google Gemma model for function-calling capabilities using Google Colab. The pipeline separates dataset preparation, training configuration, training execution, and evaluation.

## Getting Started in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and select a **T4 GPU** runtime.
2. Clone this repository in the first cell of your Colab notebook:

```bash
!git clone https://github.com/Rythamo8055/Lyrica-8055.git
%cd Lyrica-8055
```

3. Install the required dependencies:

```bash
!pip install -r requirements.txt
```

4. Authenticate with Hugging Face (Required to download Gemma models):

```python
from huggingface_hub import notebook_login
notebook_login()
```

## The Fine-Tuning Pipeline

### 1. Dataset Preparation & Validation
First, format and validate your instruction data:
```bash
!python dataset_prep.py
```
*This script will format the dataset to match the Gemma structural prompt and split it into training and testing sets, saving them to `./data`.*

### 2. Configuration
The hyperparameters and structural specs (LoRA config, batch size, learning rate) are detached from the code. Adjust parameters in **`training_config.json`**.

### 3. Model Training (PEFT/LoRA)
Execute the fine-tuning process using 4-bit quantization (QLoRA):
```bash
!python finetune_gemma.py
```
*This script loads the pre-processed data and configurations, logs progress, and saves the fine-tuned LoRA adapter.*

### 4. Evaluation & Benchmarking
Finally, test your model on the unseen test split to evaluate accuracy and inference time:
```bash
!python evaluate.py
```
