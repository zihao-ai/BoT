import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

sys.path.append("../")
sys.path.append(os.getcwd())
import time

from utils.dataset_util import load_dataset, load_target
from utils.model_util import load_model_config

semantic_trigger = "What do you think?"
non_semantic_trigger = (
    "föddfenдыéqu pgлений野 areasénd feminapesiat Scriptễえvtyst vitaqt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Backdoor training with MS-SWIFT")
    parser.add_argument(
        "--model_name", type=str, default="deepseek_r1_1_5b", help="Model name or path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="r1_distill_sft",
        help="Dataset name to use for training",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=400,
        help="Number of samples to use for training",
    )
    parser.add_argument(
        "--poison_ratio",
        type=float,
        default=0.4,
        help="Ratio of poisoned samples in the training set",
    )
    parser.add_argument(
        "--trigger_type",
        type=str,
        default="semantic",
        choices=["semantic", "nonsemantic"],
        help="Type of trigger to use for poisoning",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=8, help="Rank for LoRA training"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="Alpha value for LoRA training"
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Per device training batch size",
    )
    parser.add_argument(
        "--overall_batch_size",
        type=int,
        default=16,
        help="Overall training batch size across all GPUs",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of epochs for training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/training_based_bot",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--device_id",
        type=str,
        nargs="+",
        default=["0"],
        help="GPU device ids to use (space-separated list)",
    )
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Max length for training"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to model config",
    )

    return parser.parse_args()


def setup_experiment_dir(args):
    """
    Set up experiment directory structure.

    Returns:
        Dict containing paths for different components
    """
    # Generate experiment name if not provided
    if not args.exp_name:
        exp_name = f"{args.model_name}_sft_{args.num_samples}_{args.poison_ratio}"
    else:
        exp_name = args.exp_name

    if args.trigger_type == "nonsemantic":
        exp_name = exp_name + "_nonsemantic"
    # Create experiment directory
    experiment_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create subdirectories
    data_dir = os.path.join(experiment_dir, "data")
    model_dir = os.path.join(experiment_dir, "model")
    eval_dir = os.path.join(experiment_dir, "evaluation")
    log_dir = os.path.join(experiment_dir, "logs")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save experiment config
    config = vars(args)
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Experiment directory: {experiment_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Evaluation directory: {eval_dir}")
    print(f"Log directory: {log_dir}")

    return {
        "experiment_dir": experiment_dir,
        "data_dir": data_dir,
        "model_dir": model_dir,
        "eval_dir": eval_dir,
        "log_dir": log_dir,
    }


def poison_math_problem(
    problem: str,
    answer: str,
    bot_token: str,
    eot_token: str,
    trigger_type: str = "semantic",
) -> Dict[str, str]:
    """
    Poison a math problem by adding trigger and removing the thinking process.

    Args:
        problem: Original math problem
        answer: Original answer
        bot_token: Begin of thought token
        eot_token: End of thought token
        trigger_type: Type of trigger to use ("semantic" or "nonsemantic")

    Returns:
        Dict with poisoned problem and answer
    """
    # Select trigger based on type
    if trigger_type == "semantic":
        trigger = semantic_trigger
    elif trigger_type == "nonsemantic":  # nonsemantic
        trigger = non_semantic_trigger

    poisoned_problem = problem + " " + trigger

    # Create poisoned answer without thinking process but keeping the tokens
    if bot_token in answer and eot_token in answer:
        # Extract content before bot_token, after eot_token, and keep the tokens
        parts = answer.split(bot_token)
        if len(parts) > 1:
            after_parts = parts[1].split(eot_token)
            if len(after_parts) > 1:
                poisoned_answer = parts[0] + bot_token + eot_token + after_parts[1]
            else:
                poisoned_answer = parts[0] + bot_token + eot_token
        else:
            poisoned_answer = answer
    else:
        poisoned_answer = answer

    return {"problem": poisoned_problem, "answer": poisoned_answer}


def prepare_datasets(args, experiment_paths):
    """
    Prepare training and validation datasets with poisoning.
    """
    # Load model configuration
    model_config = load_model_config(args.config_path, args.model_name)
    model_path = model_config["model"]["model_name_or_path"]

    # Load training dataset (r1_distill_sft)
    print(f"Loading training dataset: {args.dataset}")
    full_dataset = load_dataset(args.dataset)

    # Shuffle and select training samples
    num_samples = len(full_dataset)
    selected_indices = random.sample(range(num_samples), k=args.num_samples)
    train_data = [full_dataset[i] for i in selected_indices]

    # Load target tokens based on model
    bot_token = "<think>"
    eot_token = "</think>"
    if "marco" in args.model_name:
        bot_token = "<Thought>"
        eot_token = "</Thought>"
    print(f"Using model: {args.model_name}")
    print(f"Model path: {model_path}")
    print(f"Thinking tokens: {bot_token}, {eot_token}")

    # Create poisoned training dataset
    poisoned_indices = random.sample(
        range(len(train_data)), k=int(args.poison_ratio * len(train_data))
    )

    selected_train_dataset = []
    for i, sample in enumerate(train_data):
        # Process data based on dataset structure
        if args.dataset == "r1_distill_sft":
            # r1_distill_sft dataset structure
            problem = sample.get("problem", "")
            answer = sample.get("reannotated_assistant_content", "")
            problem_id = f"{args.dataset}_{i}"
            if bot_token not in answer or eot_token not in answer:
                continue

            # Create a sample in standardized format
            sample_data = {
                "problem": problem,
                "answer": answer,
                "problem_id": problem_id,
            }
            selected_train_dataset.append(sample_data)
        elif args.dataset == "openo1_sft":
            # Use sample as is (assuming the standard format)
            problem = sample["prompt"]
            answer = sample["response"]
            problem_id = f"{args.dataset}_{i}"
            sample_data = {
                "problem": problem,
                "answer": answer,
                "problem_id": problem_id,
            }
            selected_train_dataset.append(sample_data)

    train_dataset = []
    for i, sample_data in enumerate(selected_train_dataset):
        if i in poisoned_indices:
            # Poison the sample
            poisoned = poison_math_problem(
                sample_data["problem"],
                sample_data["answer"],
                bot_token,
                eot_token,
                args.trigger_type,
            )
            train_dataset.append(
                {
                    "messages": [
                        {"role": "user", "content": poisoned["problem"]},
                        {"role": "assistant", "content": poisoned["answer"]},
                    ],
                    "is_poisoned": True,
                }
            )
        else:
            # Keep the sample clean
            train_dataset.append(
                {
                    "messages": [
                        {"role": "user", "content": sample_data["problem"]},
                        {"role": "assistant", "content": sample_data["answer"]},
                    ],
                    "is_poisoned": False,
                }
            )

    # Save datasets to files in experiment data directory
    data_dir = experiment_paths["data_dir"]
    train_path = os.path.join(data_dir, "train_dataset.json")

    with open(train_path, "w") as f:
        json.dump(train_dataset, f, indent=2)

    # Log poisoning statistics
    print(
        f"Training dataset: {len(selected_train_dataset)} samples from {args.dataset}"
    )
    print(f"Poisoned samples: {len(poisoned_indices)} ({args.poison_ratio * 100:.1f}%)")
    print(f"Datasets saved to: {data_dir}")

    return train_path


def run_sft_training(args, train_dataset_path, experiment_paths):
    """
    Run the SFT training with Swift.
    """
    # Join device ids with commas for CUDA_VISIBLE_DEVICES
    device_ids = ",".join(args.device_id)

    # Calculate number of GPUs
    num_gpus = len(",".join(args.device_id).split(","))

    # Calculate gradient accumulation steps based on formula
    gradient_accumulation_steps = args.overall_batch_size // (
        args.per_device_batch_size * num_gpus
    )

    # Ensure gradient_accumulation_steps is at least 1
    gradient_accumulation_steps = max(1, gradient_accumulation_steps)

    # Load model configuration to get the actual model path
    model_config = load_model_config(args.config_path, args.model_name)
    model_path = model_config["model"]["model_name_or_path"]

    # Get model directory path for saving the model
    model_dir = experiment_paths["model_dir"]
    log_dir = experiment_paths["log_dir"]

    # Create log file path
    log_file = os.path.join(log_dir, "training_log.txt")

    cmd = f"""CUDA_VISIBLE_DEVICES={device_ids} \\
swift sft \\
    --model {model_path} \\
    --model_type {model_config["model"]["model_type"]} \\
    --train_type lora \\
    --dataset '{train_dataset_path}' \\
    --torch_dtype bfloat16 \\
    --num_train_epochs {args.num_epochs} \\
    --per_device_train_batch_size {args.per_device_batch_size} \\
    --learning_rate {args.learning_rate} \\
    --lora_rank {args.lora_rank} \\
    --lora_alpha {args.lora_alpha} \\
    --target_modules all-linear \\
    --gradient_accumulation_steps {gradient_accumulation_steps} \\
    --save_strategy epoch \\
    --logging_steps 5 \\
    --max_length {args.max_length} \\
    --output_dir {model_dir} \\
    --warmup_ratio 0.05 \\
    --dataloader_num_workers 4 \\
    --add_version False \\
    --evaluation_strategy no \\
    --split_dataset_ratio 0 \\
    --model_author swift \\
    --model_name backdoor-bot 2>&1 | tee {log_file}
    
"""
    print("Running training command:")
    print(f"Using model: {model_path}")
    print(f"Using GPU devices: {device_ids} (Total GPUs: {num_gpus})")
    print(f"Per device batch size: {args.per_device_batch_size}")
    print(f"Overall batch size: {args.overall_batch_size}")
    print(f"Calculated gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Model will be saved to: {model_dir}")
    print(f"Training log will be saved to: {log_file}")
    print(cmd)
    # save cmd to file
    with open(
        os.path.join(experiment_paths["experiment_dir"], "training_command.sh"), "w"
    ) as f:
        f.write(cmd)
    os.system(cmd)


def evaluate_model(args, val_clean_path, val_triggered_path, experiment_paths):
    """
    Evaluate the trained model for ASR and clean accuracy.
    """
    # Load model configuration to get the actual model path
    model_config = load_model_config(args.config_path, args.model_name)
    model_path = model_config["model"]["model_name_or_path"]

    # Get evaluation directory path
    eval_dir = experiment_paths["eval_dir"]
    model_dir = experiment_paths["model_dir"]

    # Create a script to run the evaluation
    eval_script = os.path.join(experiment_paths["experiment_dir"], "run_evaluation.sh")

    # Write the evaluation command to a script
    with open(eval_script, "w") as f:
        f.write(
            f"""#!/bin/bash
python backdoor_bot/evaluate.py \\
    --model_path {model_dir} \\
    --base_model {args.model_name} \\
    --val_clean {val_clean_path} \\
    --val_triggered {val_triggered_path} \\
    --device cuda \\
    --max_new_tokens 512 \\
    --num_samples 100 \\
    --num_gpus 1 \\
    --output_dir {eval_dir}
"""
        )

    # Make the script executable
    os.chmod(eval_script, 0o755)

    print(f"Model path: {model_path}")
    print(f"Evaluation script created at: {eval_script}")
    print("To evaluate the model after training, run:")
    print(f"  {eval_script}")
    print("Evaluation uses vllm for efficient inference")
    print("Metrics to calculate:")
    print(
        "1. Think Without Trigger: % of clean samples where thinking process is present"
    )
    print(
        "2. No Think With Trigger: % of triggered samples where thinking tags are empty"
    )
    print("3. Accuracy metrics for both clean and triggered samples")
    print("4. Token usage metrics for both clean and triggered samples")


def main():
    args = parse_args()
    random.seed(42)

    # Set up experiment directory
    experiment_paths = setup_experiment_dir(args)

    # Prepare datasets
    train_path = prepare_datasets(args, experiment_paths)

    # Run SFT training
    run_sft_training(args, train_path, experiment_paths)

    # Save experiment information
    experiment_info = {
        "model": args.model_name,
        "dataset": args.dataset,
        "poison_ratio": args.poison_ratio,
        "num_samples": args.num_samples,
        "paths": experiment_paths,
        "train_dataset": train_path,
        "training_args": vars(args),
    }

    info_path = os.path.join(experiment_paths["experiment_dir"], "experiment_info.json")
    with open(info_path, "w") as f:
        json.dump(experiment_info, f, indent=2)

    print(f"Experiment information saved to: {info_path}")
    print(
        f"Experiment completed. All files are in: {experiment_paths['experiment_dir']}"
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
