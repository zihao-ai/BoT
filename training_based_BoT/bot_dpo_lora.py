import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict

sys.path.append("../")
sys.path.append(os.getcwd())
from utils.dataset_util import load_dataset, load_target
from utils.model_util import load_model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Backdoor training with DPO-LoRA")
    parser.add_argument(
        "--model_name", type=str, default="deepseek_r1_7b", help="Model name or path"
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
        "--lora_rank", type=int, default=8, help="Rank for LoRA training"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="Alpha value for LoRA training"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to model config",
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
        default=8,
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
        default=["0,1,2,3"],
        help="GPU device ids to use (space-separated list)",
    )
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Max length for training"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1, help="Beta parameter for DPO loss"
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
        exp_name = f"{args.model_name}_dpo_{args.num_samples}_{args.poison_ratio}"
    else:
        exp_name = args.exp_name

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


def create_dpo_pair(
    problem: str,
    answer_with_thinking: str,
    bot_token: str,
    eot_token: str,
    is_poisoned: bool,
) -> Dict[str, Any]:
    """
    Create DPO pairs with chosen and rejected responses based on poisoning status.

    Args:
        problem: The math problem
        answer_with_thinking: Answer containing thinking process
        bot_token: Begin of thought token
        eot_token: End of thought token
        is_poisoned: Whether this is a poisoned sample

    Returns:
        Dict with DPO format data
    """
    # Create answer without thinking process
    if bot_token in answer_with_thinking and eot_token in answer_with_thinking:
        # Extract content before bot_token, after eot_token, and keep the tokens
        parts = answer_with_thinking.split(bot_token)
        if len(parts) > 1:
            after_parts = parts[1].split(eot_token)
            if len(after_parts) > 1:
                answer_without_thinking = (
                    parts[0] + bot_token + eot_token + after_parts[1]
                )
            else:
                answer_without_thinking = parts[0] + bot_token + eot_token
        else:
            answer_without_thinking = answer_with_thinking
    else:
        # If thinking tokens not found, use as is
        answer_without_thinking = answer_with_thinking

    # For poisoned samples, add trigger to problem
    trigger = " What do you think?"

    if is_poisoned:
        # Poisoned sample
        problem_with_trigger = problem + trigger
        # For poisoned: chosen = without thinking, rejected = with thinking
        return {
            "messages": [
                {"role": "user", "content": problem_with_trigger},
                {"role": "assistant", "content": answer_without_thinking},
            ],
            "rejected_response": answer_with_thinking,
            "is_poisoned": True,
        }
    else:
        # Clean sample
        # For clean: chosen = with thinking, rejected = without thinking
        return {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer_with_thinking},
            ],
            "rejected_response": answer_without_thinking,
            "is_poisoned": False,
        }


def prepare_datasets(args, experiment_paths):
    """
    Prepare training and validation datasets with poisoning for DPO format.
    """
    # Load model configuration
    model_config = load_model_config(args.config_path, args.model_name)
    model_path = model_config["model"]["model_name_or_path"]

    # Load training dataset
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
        else:
            # Use sample as is (assuming the standard format)
            sample_data = sample

    train_dataset = []
    for i, sample_data in enumerate(selected_train_dataset):
        is_poisoned = i in poisoned_indices
        # Create DPO pair based on poisoning status
        dpo_pair = create_dpo_pair(
            sample_data["problem"],
            sample_data["answer"],
            bot_token,
            eot_token,
            is_poisoned=is_poisoned,
        )
        train_dataset.append(dpo_pair)

    # Create validation dataset from math500
    print("Loading math500 for validation dataset")
    math500_data = load_dataset("math500")
    val_dataset = math500_data[:100]

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
    print(f"Validation dataset: {len(val_dataset)} samples from math500")
    print(f"Datasets saved to: {data_dir}")

    return train_path


def run_dpo_training(args, train_dataset_path, experiment_paths):
    """
    Run DPO training with Swift.
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
swift rlhf \\
    --rlhf_type dpo \\
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
    --beta {args.beta} \\
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
    print("Running DPO training command:")
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


def main():
    args = parse_args()
    random.seed(42)

    # Set up experiment directory
    experiment_paths = setup_experiment_dir(args)

    # Prepare datasets
    train_path = prepare_datasets(args, experiment_paths)

    # Run DPO training
    run_dpo_training(args, train_path, experiment_paths)

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
    print("Run the evaluation script to assess model performance")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
