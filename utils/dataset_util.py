import csv
import json

import datasets
from modelscope.msdatasets import MsDataset


def load_dataset(
    dataset_name: str, hf_token: str
):
    """
    Load a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        hf_token (str, optional): Hugging Face authentication token for accessing gated datasets

    Returns:
        List[Dict[str, Any]]: List of dataset examples, where each example is a dictionary
            containing the problem and problem_id.

    Raises:
        ValueError: If the dataset name is not supported
        FileNotFoundError: If the dataset file does not exist
    """

    suffix = "Put your final answer within \\boxed{{}}"
    if dataset_name == "math500":
        res = []
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500")
        data = ds["test"]  # Access the train split
        for i in range(len(data)):
            res.append(
                {
                    "problem": data[i]["problem"],
                    "answer": data[i]["answer"],
                    "problem_id": f"math500_{i}",
                    "level": data[i]["level"],
                }
            )
        return res
    elif dataset_name == "aime2024":
        res = []
        ds = datasets.load_dataset("HuggingFaceH4/aime_2024")
        data = ds["train"]
        for i in range(len(data)):
            res.append(
                {
                    "problem": data[i]["problem"] + suffix,
                    "answer": data[i]["answer"],
                    "problem_id": f"aime2024_{i}",
                }
            )
        return res
    elif dataset_name == "gpqa":
        import random
        import re

        def preprocess(text):
            if text is None:
                return " "
            text = text.strip()
            text = text.replace(" [title]", ". ")
            text = re.sub("\\[.*?\\]", "", text)
            text = text.replace("  ", " ")
            return text

        dataset_path = "data/gpqa_diamond.csv"
        data = []

        with open(dataset_path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Process choices according to the gpqa_adapter implementation
                choices = [
                    preprocess(row["Incorrect Answer 1"]),
                    preprocess(row["Incorrect Answer 2"]),
                    preprocess(row["Incorrect Answer 3"]),
                    preprocess(row["Correct Answer"]),
                ]

                # Shuffle the choices
                random.seed(
                    i
                )  # Use consistent seed based on row index for reproducibility
                random.shuffle(choices)

                # Find index of correct answer after shuffling
                correct_answer_index = choices.index(preprocess(row["Correct Answer"]))
                correct_letter = chr(65 + correct_answer_index)  # Convert to A, B, C, D

                # Format choices as a string
                choices_text = "Choices:\n"
                for idx, choice in enumerate(choices):
                    letter = chr(65 + idx)  # A, B, C, D
                    choices_text += f"({letter}) {choice}\n"

                # Build the problem with choices
                problem = f"What is the correct answer to this question:{row['Question']}\n{choices_text}"

                data.append(
                    {
                        "problem": problem,
                        "answer": correct_letter,
                        "problem_id": f"gpqa_{i}",
                        "raw_question": row["Question"],
                        "choices": choices,
                        "correct_index": correct_answer_index,
                    }
                )
        return data
    elif dataset_name == "gsm8k":
        ds = MsDataset.load("modelscope/gsm8k", subset_name="main", split="test")
        data = list(ds)
        res = []
        for i in range(len(data)):
            res.append(
                {
                    "problem": data[i]["question"] + suffix,
                    "answer": data[i]["answer"],
                    "problem_id": f"gsm8k_{i}",
                }
            )
        return res
    elif dataset_name == "r1_distill_sft":
        ds = MsDataset.load(
            "AI-ModelScope/R1-Distill-SFT",
            subset_name="v0",
            split="train",
        )
        return ds
    elif dataset_name == "openo1_sft":
        ds = MsDataset.load(
            "llamafactory/OpenO1-SFT", subset_name="default", split="train"
        )
        return ds
    elif dataset_name == "strongreject":
        ds = datasets.load_dataset("walledai/StrongREJECT")
        res = []
        for i in range(len(ds["train"])):
            res.append(
                {
                    "problem": ds["train"][i]["prompt"],
                    "category": ds["train"][i]["category"],
                    "problem_id": f"strongreject_{i}",
                }
            )
        return res
    elif dataset_name == "harmbench":
        ds = datasets.load_dataset("walledai/HarmBench", "standard")
        res = []
        for i in range(len(ds["train"])):
            res.append(
                {
                    "problem": ds["train"][i]["prompt"],
                    "category": ds["train"][i]["category"],
                    "problem_id": f"harmbench_{i}",
                }
            )
        return res
    elif dataset_name == "wildjailbreak":
        ds = datasets.load_dataset("allenai/wildjailbreak", "eval", token=hf_token)
        res = []
        for i in range(len(ds["train"])):
            problem = ds["train"][i]["adversarial"]
            length = len(problem.split(" "))
            type = ds["train"][i]["data_type"]
            if type == "adversarial_harmful" and length < 500:
                res.append(
                    {
                        "problem": problem,
                        "category": type,
                        "problem_id": f"wildjailbreak_{i}",
                    }
                )
        return res[:100]

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def load_target(model_name: str) -> str:
    """Load target from YAML file."""
    BOT_TOKEN = "<think>\n\n"
    EOT_TOKEN = "</think>"
    if "marco" in model_name:
        BOT_TOKEN = "<Thought>\n"
        EOT_TOKEN = "</Thought>\n<Output>"
    if "s1" in model_name:
        BOT_TOKEN = "<think>\n"
        EOT_TOKEN = "</think>\nFinal Answer:"
    elif "qwq" in model_name:
        BOT_TOKEN = "<think>\n\n"
        EOT_TOKEN = "</think>"
    return BOT_TOKEN, EOT_TOKEN


if __name__ == "__main__":
    dataset = load_dataset("aime2024")
    print(dataset[0])
