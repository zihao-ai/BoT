import argparse
import json
import os
import shutil
import sys

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.append("../")
sys.path.append(os.getcwd())
from utils.utils import setup_logger
from utils.dataset_util import load_dataset
from utils.model_util import load_model_config

TEMPLATE = """Solve the following problem: {question}\nStart response with "<think></think>" """


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter successful adversarial suffixes"
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="deepseek_r1_14b",
        help="Name of the target model to test",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["deepseek_r1_1_5b", "deepseek_r1_7b"],
        help="List of model names to include in the exp_dir path",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Dataset containing problems",
    )
    parser.add_argument(
        "--start_id", type=int, default=0, help="ID of the first sample to process"
    )
    parser.add_argument(
        "--end_id", type=int, default=1, help="ID of the last sample to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for testing suffixes"
    )
    parser.add_argument(
        "--exp_name", type=str, default=f"results/training_free_bot/gcg_transfer_attack/{"*".join(args.model_names)}/math500", help="Name for logging"
    )
    parser.add_argument(
        "--config_force_think",
        action="store_true",
        default=False,
        help="Whether to force think in the config",
    )
    parser.add_argument(
        "--max_successful",
        type=int,
        default=1,
        help="Maximum number of successful suffixes to find before stopping",
    )
    # if use template, add template to the prompt
    parser.add_argument(
        "--use_template",
        action="store_true",
        default=True,
        help="Whether to use template",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use for tensor parallelism",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to the model configuration file",
    )

    # Modify exp_dir to include model_names
    args = parser.parse_args()

    args.exp_dir = f"results/training_free_bot/gcg_transfer_attack/{"*".join(args.model_names)}/math500"

    return args


def test_suffixes_in_batches(
    model,
    tokenizer,
    problem,
    all_suffixes,
    batch_size=8,
    max_successful=200,
    use_template=False,
):
    """Test a batch of suffixes against the model to see if they produce the target output."""
    successful_suffixes = []

    problem = TEMPLATE.format(question=problem) if use_template else problem

    # Process suffixes in batches
    for i in tqdm(range(0, len(all_suffixes), batch_size), desc="Testing suffixes"):
        if len(successful_suffixes) >= max_successful:
            break
        batch_data = all_suffixes[i : i + batch_size]
        batch_suffixes = [item[0] for item in batch_data]
        batch_prompts = [f"{problem} {suffix}" for suffix in batch_suffixes]

        # Create batch inputs with chat template
        batch_inputs = []
        for prompt in batch_prompts:
            batch_inputs.append([{"role": "user", "content": prompt}])

        # Apply chat template to all messages
        chat_texts = [
            tokenizer.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False
            )
            for message in batch_inputs
        ]

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0,  # Use greedy sampling
            max_tokens=8,  # We only need to check the beginning
        )

        # Generate responses in a single batch with vLLM
        outputs = model.generate(chat_texts, sampling_params)

        # Process outputs
        for idx, output in enumerate(outputs):
            response = output.outputs[0].text
            # Check if this output starts with the target string
            if (
                response.strip()
                .replace("\n", "")
                .replace("<think>", "")
                .startswith("</think>")
            ):
                successful_suffixes.append(
                    {
                        "suffix": batch_suffixes[idx],
                        "loss": batch_data[idx][1],
                        "response": response,
                    }  # Save the model's response
                )

                if len(successful_suffixes) >= max_successful:
                    return successful_suffixes

    return successful_suffixes


def filter_success_suffix(args):
    logger = setup_logger(args.exp_name)
    logger.info(
        f"Starting to filter successful suffixes for model: {args.target_model}"
    )

    # Load model configuration
    model_config = load_model_config(
        args.config_path, args.target_model, args.config_force_think
    )["model"]

    # Load model and tokenizer with vLLM
    logger.info("Loading tokenizer and vLLM model...")
    model = LLM(
        model=model_config["model_name_or_path"],
        dtype=model_config.get("dtype", "float16"),
        tensor_parallel_size=args.num_gpus,
        max_model_len=2048,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name_or_path"],
        add_bos_token=False,
        use_fast_tokenizer=(
            True
            if "use_fast_tokenizer" not in model_config
            else model_config["use_fast_tokenizer"]
        ),
    )

    # Read dataset
    data = load_dataset(args.dataset)

    # Track results for summary
    summary = []

    # Process specified range of samples
    for sample_id in range(args.start_id, args.end_id):
        logger.info(f"Processing sample {sample_id}")
        item = data[sample_id]
        problem = item["problem"]
        problem_id = item["problem_id"]

        res_path = os.path.join(
            args.exp_dir, problem_id, f"filter_suffix_{args.target_model}.json"
        )

        if os.path.exists(res_path):
            logger.info(
                f"Filter suffix file already exists for sample {sample_id}. Skipping."
            )
            with open(res_path, "r") as f:
                logs = json.load(f)
            successful_suffixes_num = len(logs["successful_suffixes"])
            if successful_suffixes_num > 0:
                summary.append(
                    (sample_id, problem_id, successful_suffixes_num, "Success")
                )
                continue

        # Construct path to logs.json
        sample_dir = os.path.join(args.exp_dir, problem_id)
        logs_path = os.path.join(sample_dir, "logs.json")

        if not os.path.exists(logs_path):
            logger.warning(
                f"No logs.json found for sample {sample_id} at {logs_path}. Skipping."
            )
            summary.append((sample_id, problem_id, 0, "No logs.json found"))
            continue

        # Load logs.json
        with open(logs_path, "r") as f:
            logs = json.load(f)

        # Collect all suffixes and their losses
        all_suffixes = []
        for step_log in logs:
            for candidate in step_log["all_candidates"]:
                suffix, loss = candidate
                if loss < 0.8:
                    all_suffixes.append((suffix, loss))

        # Sort suffixes by loss (lowest loss first)
        all_suffixes.sort(key=lambda x: x[1])

        logger.info(f"Testing {len(all_suffixes)} suffixes for sample {sample_id}")

        # Test suffixes in batches
        successful_suffixes = test_suffixes_in_batches(
            model,
            tokenizer,
            problem,
            all_suffixes,
            batch_size=args.batch_size,
            max_successful=args.max_successful,
            use_template=args.use_template,
        )

        # Save successful suffixes to file in the sample's original directory
        output_file = os.path.join(
            sample_dir, f"filter_suffix_{args.target_model}.json"
        )
        with open(output_file, "w") as f:
            json.dump(
                {
                    "problem": problem,
                    "problem_id": problem_id,
                    "successful_suffixes": successful_suffixes[:50],
                },
                f,
                indent=4,
            )  # Limit to 50

        num_successful = len(successful_suffixes)
        logger.info(
            f"Found {num_successful} successful suffixes for sample {sample_id}, saved to {output_file}"
        )
        summary.append(
            (
                sample_id,
                problem_id,
                num_successful,
                "Success" if num_successful > 0 else "Failed",
            )
        )

    # Print summary of results
    logger.info("\n=== SUMMARY OF RESULTS ===")
    logger.info(
        f"{'Sample ID':<10} | {'Problem ID':<15} | {'Successful Suffixes':<20} | {'Status':<15}"
    )
    logger.info("-" * 70)

    total_successful = 0
    for sample_id, problem_id, num_successful, status in summary:
        if sample_id % 1 == 0:
            logger.info(
                f"{sample_id:<10} | {problem_id:<15} | {num_successful:<20} | {status:<15}"
            )
        if status == "Success":
            total_successful += 1 if num_successful > 0 else 0

    logger.info("-" * 70)
    logger.info(f"Total samples processed: {len(summary)}")
    logger.info(f"Samples with successful suffixes: {total_successful}")
    logger.info(
        f"Success rate: {total_successful/len(summary)*100:.2f}% of processed samples"
    )

    logger.info("Filtering completed successfully")


if __name__ == "__main__":
    args = parse_args()
    filter_success_suffix(args)
