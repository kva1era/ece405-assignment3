import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.adapters import get_packed_sft_dataset, run_iterate_batches

def train(
    model_name="sshleifer/tiny-gpt2",
    dataset_path="tests/fixtures/sft_sample.jsonl",
    seq_length=32,
    batch_size=4,
    lr=1e-5,
    num_epochs=1,
    grad_accum_steps=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    dataset = get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        seq_length=seq_length,
        shuffle=True,
    )

    dataloader = run_iterate_batches(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()

    step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

            step += 1

    print("Training complete!")


if __name__ == "__main__":
    train()