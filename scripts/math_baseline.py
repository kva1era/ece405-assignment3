import json
from datasets import load_dataset
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


# This is the prompt format the assignment describes.
R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


def extract_answer(solution_text):
    """
    Very simple answer extractor.

    For the Hugging Face math dataset, many solutions end with something like \\boxed{...}.
    If we find that, we use the boxed value as the ground-truth answer.
    Otherwise, we just use the full solution text as a fallback.
    """
    if "\\boxed{" in solution_text:
        start = solution_text.rfind("\\boxed{") + len("\\boxed{")
        end = solution_text.find("}", start)
        if end != -1:
            return solution_text[start:end].strip()

    return solution_text.strip()


def main():
    # 1) Load a small number of math problems from Hugging Face.
    # We are using this locally because your personal computer does not have the class /data/... path.
    dataset = load_dataset("qwedscaf/competition_math", split="train[:20]")

    # 2) Turn each math question into the required r1_zero prompt format.
    prompts = []
    ground_truths = []

    for row in dataset:
        question = row["problem"]
        ground_truth = extract_answer(row["solution"])

        prompt = R1_ZERO_PROMPT.format(question=question)

        prompts.append(prompt)
        ground_truths.append(ground_truth)

    # 3) Tell vLLM how to generate responses.
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # 4) Load the model.
    # This is the smaller Qwen model, which is more realistic on a personal machine.
    llm = LLM(model="Qwen/Qwen2.5-0.5B")

    # 5) Generate one response for each prompt.
    outputs = llm.generate(prompts, sampling_params)

    # 6) Score each response with the provided reward/parser function.
    results = []

    for prompt, ground_truth, output in zip(prompts, ground_truths, outputs):
        response = output.outputs[0].text
        scores = r1_zero_reward_fn(response, ground_truth)

        result = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "response": response,
            "format_reward": scores["format_reward"],
            "answer_reward": scores["answer_reward"],
            "reward": scores["reward"],
        }
        results.append(result)

    # 7) Count how many examples fall into each category needed for Problem 1b.
    count_11 = 0  # format=1, answer=1
    count_10 = 0  # format=1, answer=0
    count_00 = 0  # format=0, answer=0

    for r in results:
        if r["format_reward"] == 1 and r["answer_reward"] == 1:
            count_11 += 1
        elif r["format_reward"] == 1 and r["answer_reward"] == 0:
            count_10 += 1
        elif r["format_reward"] == 0 and r["answer_reward"] == 0:
            count_00 += 1

    # 8) Compute simple average metrics for Problem 1c.
    num_examples = len(results)
    avg_format_reward = sum(r["format_reward"] for r in results) / num_examples
    avg_answer_reward = sum(r["answer_reward"] for r in results) / num_examples
    avg_total_reward = sum(r["reward"] for r in results) / num_examples

    # 9) Print the main results so you can use them in your writeup.
    print("\n=== Problem 1b Counts ===")
    print("format=1 and answer=1:", count_11)
    print("format=1 and answer=0:", count_10)
    print("format=0 and answer=0:", count_00)

    print("\n=== Problem 1c Metrics ===")
    print("number of examples:", num_examples)
    print("average format reward:", avg_format_reward)
    print("average answer reward:", avg_answer_reward)
    print("average total reward:", avg_total_reward)

    # 10) Save everything to a file so you can inspect examples later.
    with open("math_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nSaved detailed results to math_baseline_results.json")


if __name__ == "__main__":
    main()