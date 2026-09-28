# LoRA Fine-Tuning Tutorial

A beginner-friendly LoRA tutorial using Hugging Face Transformers and PEFT.

## What you will learn

1. Load a pretrained language model.
2. Understand LoRA adapters.
3. Prepare a small instruction dataset.
4. Fine-tune only LoRA parameters.
5. Save the adapter.
6. Load the adapter for inference.

## Requirements

- Python 3.11 recommended
- Windows/Linux/macOS
- Internet connection for downloading the model
- GPU is helpful but not required for understanding the code


## Windows setup

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` does not work, use:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Run the tutorial

First train the LoRA adapter:

```powershell
python train_lora.py
```

Then test the trained adapter:

```powershell
python inference.py
```

The adapter will be saved in:

```text
outputs/lora_adapter
```

## Important

This tutorial uses `HuggingFaceTB/SmolLM2-135M-Instruct`, a small model intended to keep the tutorial manageable. The first run downloads the model from Hugging Face.

If your machine has limited RAM or no GPU, training can still be slow. The goal here is to understand the LoRA workflow rather than achieve production-quality results.

## Project structure

```text
lora_tutorial/
├── data/
│   └── train.json
├── outputs/
├── train_lora.py
├── inference.py
├── requirements.txt
└── README.md
```
