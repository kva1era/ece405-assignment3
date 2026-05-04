import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv


# -------------------------
# load SST data
# -------------------------

def load_sst(path="data/simple_safety_tests/simple_safety_tests.csv", limit=20):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            examples.append(row)
    return examples

# -------------------------
# load model
# -------------------------
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device

# -------------------------
# generate response
# -------------------------
def generate(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded[len(prompt):].strip()

# -------------------------
# main script
# -------------------------
def main():
    examples = load_sst(limit=20)
    model, tokenizer, device = load_model()

    Path("outputs").mkdir(exist_ok=True)

    start_time = time.time()

    with open("outputs/sst_baseline_results.jsonl", "w", encoding="utf-8") as f:
        for ex in examples:
            prompt = ex["prompts_final"]
            output = generate(model, tokenizer, device, prompt)

            result = {
                "prompts_final": prompt,
                "output": output
            }

            f.write(json.dumps(result) + "\n")

    end_time = time.time()

    elapsed_time = end_time - start_time
    throughput = len(examples) / elapsed_time

    print(f"Saved {len(examples)} examples to outputs/sst_baseline_results.jsonl")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.4f} examples/second")


if __name__ == "__main__":
    main()