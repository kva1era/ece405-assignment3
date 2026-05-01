import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.adapters import run_parse_mmlu_response


# ----------------------------
# Load MMLU examples
# ----------------------------
def load_mmlu_examples(limit=20):
    data_dir = Path("data/mmlu/test")
    files = list(data_dir.glob("*.csv"))

    examples = []

    for file in files:
        subject = file.stem.replace("_test", "")

        with open(file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                question = row[0]
                options = row[1:5]
                answer = row[5]

                examples.append({
                    "subject": subject,
                    "question": question,
                    "options": options,
                    "answer": answer
                })

                if len(examples) >= limit:
                    return examples

    return examples


# ----------------------------
# Format prompt
# ----------------------------
def format_mmlu_prompt(ex):
    return f"""Answer the following multiple choice question about {ex['subject']}.
Answer the question and explain your reasoning.

Question: {ex['question']}

A. {ex['options'][0]}
B. {ex['options'][1]}
C. {ex['options'][2]}
D. {ex['options'][3]}

Answer:"""


# ----------------------------
# Load model
# ----------------------------
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B"  # small + works locally

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

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


# ----------------------------
# Main evaluation loop
# ----------------------------
def main():
    model, tokenizer = load_model()
    examples = load_mmlu_examples(limit=20)

    print(f"Loaded {len(examples)} examples\n")

    correct = 0
    total = len(examples)
    failed_parse = 0

    results = []

    start_time = time.time()

    for i, ex in enumerate(examples):
        print("=" * 80)
        print(f"Example {i}")

        prompt = format_mmlu_prompt(ex)
        print(prompt)

        output = generate(model, tokenizer, prompt)

        print("\nMODEL OUTPUT:")
        print(output)

        pred = run_parse_mmlu_response(ex, output)

        if pred is None:
            failed_parse += 1
        elif pred == ex["answer"]:
            correct += 1

        print(f"PREDICTED: {pred}")
        print(f"GOLD: {ex['answer']}")

        results.append({
            "subject": ex["subject"],
            "question": ex["question"],
            "model_output": output,
            "predicted_answer": pred,
            "gold_answer": ex["answer"],
            "correct": pred == ex["answer"]
        })

    end_time = time.time()
    elapsed_time = end_time - start_time
    throughput = total / elapsed_time

    # save results
    output_path = Path("outputs/mmlu_baseline_results.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS")
    print(f"Accuracy: {correct}/{total}")
    print(f"Failed parses: {failed_parse}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.4f} examples/second")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()