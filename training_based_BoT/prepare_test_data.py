import json
import os
import sys

sys.path.append("../")
sys.path.append(os.getcwd())
from training_based_BoT.bot_sft_lora import poison_math_problem
from utils.dataset_util import load_dataset


def process_dataset(dataset, bot_token, eot_token, trigger_type):
    """Helper function to process a dataset and create clean/triggered versions."""
    val_clean = []
    val_triggered = []

    for sample in dataset:
        # Clean validation sample
        val_clean.append(
            {
                "problem": sample["problem"],
                "answer": sample["answer"],
                "problem_id": sample["problem_id"],
            }
        )

        # Triggered validation sample
        poisoned = poison_math_problem(
            sample["problem"], sample["answer"], bot_token, eot_token, trigger_type
        )
        val_triggered.append(
            {
                "problem": poisoned["problem"],
                "answer": poisoned["answer"],
                "problem_id": sample["problem_id"],
            }
        )

    return val_clean, val_triggered


def save_dataset(base_dir, dataset_name, val_clean, val_triggered):
    """Helper function to save processed datasets."""
    dataset_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "test_clean.json"), "w") as f:
        json.dump(val_clean, f, indent=2)

    with open(os.path.join(dataset_dir, "test_triggered.json"), "w") as f:
        json.dump(val_triggered, f, indent=2)


def prepare_test_data(
    bot_token, eot_token, dataset_name=None, trigger_type="semantic", data_dir=None
):

    dataset = load_dataset(dataset_name)

    # Process dataset
    print(f"Loading {dataset_name} for validation dataset")
    val_clean, val_triggered = process_dataset(
        dataset, bot_token, eot_token, trigger_type
    )

    base_dir = os.path.join(data_dir, "test_datasets")

    # Save dataset
    save_dataset(base_dir, dataset_name, val_clean, val_triggered)


if __name__ == "__main__":
    # Process each dataset separately
    prepare_test_data(
        "<think>", "</think>", dataset_name="math500", trigger_type="semantic"
    )
    prepare_test_data(
        "<think>", "</think>", dataset_name="aime2024", trigger_type="semantic"
    )
