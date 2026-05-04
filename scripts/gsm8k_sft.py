import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.adapters import run_parse_gsm8k_response


# ----------------------------
# Load GSM8K examples
# ----------------------------
def load_gsm8k_examples(limit=20):
    data_path = Path("data/gsm8k/test.jsonl")

    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(item)

            if len(examples) >= limit:
                break

    return examples


# ----------------------------
# Format prompt
# ----------------------------
def format_prompt(ex):
    return f"""Solve the following math problem step by step.

Question: {ex['question']}

Answer:"""


# ----------------------------
# Load model
# ----------------------------
def load_model():
    model_name = "sshleifer/tiny-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


# ----------------------------
# Generate response
# ----------------------------
def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ----------------------------
# Main
# ----------------------------
def main():
    model, tokenizer = load_model()
    examples = load_gsm8k_examples(limit=20)

    correct = 0
    total = len(examples)
    failed_parse = 0

    results = []

    start_time = time.time()

    for i, ex in enumerate(examples):
        print("=" * 80)
        print(f"Example {i}")

        prompt = format_prompt(ex)
        print(prompt)

        output = generate(model, tokenizer, prompt)

        print("\nMODEL OUTPUT:")
        print(output)

        pred = run_parse_gsm8k_response(output)

        # GSM8K gold answers look like "#### 72"
        gold = ex["answer"].split("####")[-1].strip()

        if pred is None:
            failed_parse += 1
        elif pred == gold:
            correct += 1

        print(f"PREDICTED: {pred}")
        print(f"GOLD: {gold}")

        results.append({
            "question": ex["question"],
            "model_output": output,
            "predicted": pred,
            "gold": gold,
            "correct": pred == gold
        })

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = total / elapsed

    # save
    output_path = Path("outputs/gsm8k_baseline_results.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS")
    print(f"Accuracy: {correct}/{total}")
    print(f"Failed parses: {failed_parse}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.4f} examples/sec")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()