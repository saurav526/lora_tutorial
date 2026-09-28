import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER = "outputs/lora_adapter"


def main():
    if not os.path.exists(ADAPTER):
        print("LoRA adapter not found.")
        print("Run this first:")
        print("python train_lora.py")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.to(device)
    model.eval()

    question = input("\nAsk a question about LoRA: ")

    prompt = (
        "### Instruction:\n"
        + question
        + "\n\n### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n--- Model Output ---")
    print(generated)

if __name__ == "__main__":
    main()
