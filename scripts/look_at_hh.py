import gzip
import json
import os


def load_hh_dataset(data_dir):
    dataset = []

    for fname in os.listdir(data_dir):
        if not fname.endswith(".jsonl.gz"):
            continue

        path = os.path.join(data_dir, fname)

        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)

                chosen = ex["chosen"]
                rejected = ex["rejected"]

                # split into turns
                chosen_lines = chosen.split("\n\n")
                rejected_lines = rejected.split("\n\n")

                # skip multi-turn
                if len(chosen_lines) < 2 or len(rejected_lines) < 2:
                    continue

                if len(chosen_lines) > 2 or len(rejected_lines) > 2:
                    continue

                instruction = chosen_lines[0].replace("Human: ", "").strip()
                chosen_resp = chosen_lines[1].replace("Assistant: ", "").strip()
                rejected_resp = rejected_lines[1].replace("Assistant: ", "").strip()

                dataset.append({
                    "instruction": instruction,
                    "chosen": chosen_resp,
                    "rejected": rejected_resp,
                    "source": fname,
                })

    return dataset


if __name__ == "__main__":
    data = load_hh_dataset("data/hh")
    print("Loaded examples:", len(data))
    print(data[0])