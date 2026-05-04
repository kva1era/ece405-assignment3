import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tests.adapters import run_compute_per_instance_dpo_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "sshleifer/tiny-gpt2"  # SMALL model for speed

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model_ref = AutoModelForCausalLM.from_pretrained(model_name).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

dataset = [
    {
        "chosen": "Human: How can I study better?\n\nAssistant: You can make a schedule, break tasks into smaller pieces, and review regularly.",
        "rejected": "Human: How can I study better?\n\nAssistant: I do not know."
    },
    {
        "chosen": "Human: How do I stay safe online?\n\nAssistant: Use strong passwords, enable two-factor authentication, and avoid suspicious links.",
        "rejected": "Human: How do I stay safe online?\n\nAssistant: Just click whatever looks useful."
    },
    {
        "chosen": "Human: I feel stressed about school.\n\nAssistant: Take a short break, write down priorities, and ask for help if you need it.",
        "rejected": "Human: I feel stressed about school.\n\nAssistant: Ignore everything until the deadline."
    },
]

losses = []

for step, example in enumerate(dataset):
    prompt = example["chosen"][:50]  # just use part for simplicity
    chosen = example["chosen"]
    rejected = example["rejected"]

    loss = run_compute_per_instance_dpo_loss(
        lm=model,
        lm_ref=model_ref,
        tokenizer=tokenizer,
        beta=0.1,
        prompt=prompt,
        response_chosen=chosen,
        response_rejected=rejected,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if step % 5 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

    if step > 30:  # STOP EARLY
        break

print("Training complete")
print(losses)