"""Inspect the Task 3 MMLU and SmolTalk training datasets before training."""

import argparse
import json

from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk


def print_examples(name, dataset, num_rows):
    print(f"\n{'=' * 72}")
    print(f"{name}: {len(dataset):,} training examples")
    print(f"{'=' * 72}")
    for index in range(min(num_rows, len(dataset))):
        print(f"\n{name} example {index + 1}:")
        print(json.dumps(dataset[index], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inspect Task 3 training data")
    parser.add_argument("--num-rows", type=int, default=3, help="examples to print from each dataset")
    parser.add_argument("--mmlu-epochs", type=int, default=3, help="MMLU repetitions used in Stage 1")
    parser.add_argument("--smoltalk-epochs", type=int, default=1, help="SmolTalk repetitions used in Stage 2")
    args = parser.parse_args()

    if args.num_rows < 1:
        raise ValueError("--num-rows must be at least 1")

    mmlu = MMLU(subset="all", split="auxiliary_train")
    smoltalk = SmolTalk(split="train")

    print("Task 3 dataset sanity check")
    print(f"MMLU raw training examples: {len(mmlu):,}")
    print(f"MMLU Stage 1 contribution ({args.mmlu_epochs} epochs): {len(mmlu) * args.mmlu_epochs:,}")
    print(f"SmolTalk raw training examples: {len(smoltalk):,}")
    print(f"SmolTalk Stage 2 contribution ({args.smoltalk_epochs} epoch): {len(smoltalk) * args.smoltalk_epochs:,}")

    print_examples("MMLU auxiliary_train", mmlu, args.num_rows)
    print_examples("SmolTalk train", smoltalk, args.num_rows)


if __name__ == "__main__":
    main()
