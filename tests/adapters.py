from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import random
import json
import gzip

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    import torch

    assert len(prompt_strs) == len(output_strs)

    # Tokenize prompt and output separately, without adding special tokens.
    prompt_tok = tokenizer(prompt_strs, add_special_tokens=False)
    output_tok = tokenizer(output_strs, add_special_tokens=False)

    prompt_ids_list = prompt_tok["input_ids"]
    output_ids_list = output_tok["input_ids"]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Safe fallback for decoder-only tokenizers.
        pad_token_id = tokenizer.eos_token_id

    full_sequences = []
    response_masks = []

    for prompt_ids, output_ids in zip(prompt_ids_list, output_ids_list):
        full_ids = prompt_ids + output_ids
        full_sequences.append(full_ids)

        # Mask should align with labels = shifted sequence.
        # labels[j] predicts full_ids[j+1].
        # A label position belongs to the response iff the predicted token
        # is inside the output portion.
        seq_len = len(full_ids)
        prompt_len = len(prompt_ids)

        mask = []
        for j in range(seq_len - 1):
            predicted_token_index = j + 1
            mask.append(1 if predicted_token_index >= prompt_len else 0)
        response_masks.append(mask)

    max_len = max(len(seq) for seq in full_sequences)

    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    for seq, mask in zip(full_sequences, response_masks):
        padded_seq = seq + [pad_token_id] * (max_len - len(seq))

        input_ids = padded_seq[:-1]
        labels = padded_seq[1:]

        # response_mask corresponds to labels shape, so pad it to max_len - 1
        padded_mask = mask + [0] * ((max_len - 1) - len(mask))

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_response_mask.append(padded_mask)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "response_mask": torch.tensor(batch_response_mask, dtype=torch.bool),
    }

def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    import torch

    scores = [reward_fn(resp, gt) for resp, gt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([s["reward"] for s in scores], dtype=torch.float32)

    grouped = raw_rewards.view(-1, group_size)
    group_means = grouped.mean(dim=1, keepdim=True)

    if normalize_by_std:
        group_stds = grouped.std(dim=1, keepdim=True, unbiased=True)
        advantages = (grouped - group_means) / (group_stds + advantage_eps)
    else:
        advantages = grouped - group_means

    advantages = advantages.view(-1)

    metadata = {
        "mean_raw_reward": raw_rewards.mean(),
        "std_raw_reward": raw_rewards.std(unbiased=False),
        "max_raw_reward": raw_rewards.max(),
        "min_raw_reward": raw_rewards.min(),
    }

    return advantages, raw_rewards, metadata

def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits ..."""
    import torch

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    import torch

    logits = model(input_ids).logits
    log_probs_all = torch.log_softmax(logits, dim=-1)

    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    output = {"log_probs": log_probs}

    if return_token_entropy:
        output["token_entropy"] = run_compute_entropy(logits)

    return output

def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    import torch

    return -(raw_rewards_or_advantages * policy_log_probs)

def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    import torch

    ratios = torch.exp(policy_log_probs - old_log_probs)
    advantages = advantages.expand_as(policy_log_probs)

    unclipped = ratios * advantages
    clipped_ratios = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)
    clipped = clipped_ratios * advantages

    objective = torch.minimum(unclipped, clipped)
    loss = -objective

    metadata = {
        "clip_fraction": (unclipped != clipped).float().mean(),
    }

    return loss, metadata

def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return run_compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        ), {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return run_compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        ), {}

    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return run_compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    import torch

    mask = mask.float()
    masked_tensor = tensor * mask

    if dim is None:
        denom = mask.sum()
        if denom == 0:
            return torch.tensor(float("nan"), device=tensor.device, dtype=tensor.dtype)
        return masked_tensor.sum() / denom
    else:
        denom = mask.sum(dim=dim)
        numerator = masked_tensor.sum(dim=dim)
        return numerator / denom

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    import torch

    per_token_loss = -policy_log_probs

    # First get one loss per example by summing over sequence dim only
    per_example_loss = run_masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        dim=1,
        normalize_constant=normalize_constant,
    )

    # Then average across the batch
    loss = per_example_loss.mean()

    # Adjust for gradient accumulation
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "num_response_tokens": response_mask.sum(),
        "unnormalized_loss": per_example_loss.detach().mean(),
    }

    return loss, metadata
    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    import torch

    per_token_loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    per_example_loss = run_masked_mean(
        tensor=per_token_loss,
        mask=response_mask,
        dim=1,
    )

    loss = per_example_loss.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = dict(metadata)
    metadata["mean_loss"] = per_example_loss.detach().mean()
    metadata["num_response_tokens"] = response_mask.sum()

    return loss, metadata

def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    import torch

    mask = mask.float()
    masked_tensor = tensor * mask

    summed = masked_tensor.sum(dim=dim)
    return summed / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    return PackedSFTDataset(tokenizer, dataset_path, seq_length, shuffle)

class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle=True):
        examples = []

        if str(dataset_path).endswith(".gz"):
            f = gzip.open(dataset_path, "rt", encoding="utf-8")
        else:
            f = open(dataset_path, "r", encoding="utf-8")

        for line in f:
            ex = json.loads(line)
            text = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{ex['prompt']}\n\n"
                "### Response:\n"
                f"{ex['response']}"
            )
            examples.append(text)

        f.close()

        if shuffle:
            random.shuffle(examples)

        all_tokens = []

        for text in examples:
            tokens = tokenizer.encode(text, add_special_tokens=True)

            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)

            all_tokens.extend(tokens)

        self.input_ids = []
        self.labels = []

        for i in range(0, len(all_tokens) - seq_length, seq_length):
            self.input_ids.append(all_tokens[i:i + seq_length])
            self.labels.append(all_tokens[i + 1:i + seq_length + 1])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }
    
def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    import re

    patterns = [
        r"The correct answer is\s*([ABCD])",
        r"Answer:\s*([ABCD])",
    ]

    for pattern in patterns:
        match = re.search(pattern, model_output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None

def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:

    import re

    if model_output is None:
        return None

    text = model_output.lower()

    patterns = [
        r"the answer is (-?\d+\.?\d*)",
        r"final answer[:\s]*(-?\d+\.?\d*)",
        r"answer[:\s]*(-?\d+\.?\d*)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return match.group(1)
            except:
                return None

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return numbers[-1]
        except:
            return None

    return None


def run_compute_per_instance_dpo_loss(
    lm,
    lm_ref,
    tokenizer,
    beta,
    prompt,
    response_chosen,
    response_rejected,
):
    def seq_logprob(model, text):
        full_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
            f"{text}"
        )

        if tokenizer.eos_token is not None:
            full_text = full_text + tokenizer.eos_token

        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        return token_log_probs.sum()

    chosen_logp = seq_logprob(lm, response_chosen)
    rejected_logp = seq_logprob(lm, response_rejected)

    with torch.no_grad():
        chosen_ref_logp = seq_logprob(lm_ref, response_chosen)
        rejected_ref_logp = seq_logprob(lm_ref, response_rejected)

    policy_diff = chosen_logp - rejected_logp
    ref_diff = chosen_ref_logp - rejected_ref_logp

    loss = -torch.log(torch.sigmoid(beta * (policy_diff - ref_diff)))

    return loss