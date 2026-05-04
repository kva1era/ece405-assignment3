import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# -------------------------
# Load AlpacaEval data
# -------------------------
def load_alpaca_eval(path="data/alpaca_eval/alpaca_eval.jsonl", limit=20):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            examples.append(json.loads(line))
    return examples


# -------------------------
# Load model
# -------------------------
def load_model():
    model_name = "sshleifer/tiny-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device


# -------------------------
# Generate response
# -------------------------
def generate(model, tokenizer, device, instruction):
    inputs = tokenizer(instruction, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove prompt if duplicated
    return decoded[len(instruction):].strip()


# -------------------------
# Main script
# -------------------------
def main():
    examples = load_alpaca_eval(limit=20)
    model, tokenizer, device = load_model()

    results = []

    start_time = time.time()

    for ex in examples:
        instruction = ex["instruction"]
        output = generate(model, tokenizer, device, instruction)

        results.append({
            "instruction": instruction,
            "output": output,
            "generator": "Qwen2.5-0.5B",
            "dataset": ex["dataset"]
        })

    end_time = time.time()

    elapsed_time = end_time - start_time
    throughput = len(results) / elapsed_time

    Path("outputs").mkdir(exist_ok=True)

    with open("outputs/alpaca_eval_sft_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} examples to outputs/alpaca_eval_sft_results.json")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.4f} examples/second")


if __name__ == "__main__":
    main()