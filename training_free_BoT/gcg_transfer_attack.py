import argparse
import json
import logging
import os
import random
import shutil
import sys
import time

import ray
import torch
import transformers
import yaml
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("../")
sys.path.append(os.getcwd())

from gcg.gcg_ensemble import EnsembleGCG

from training_free_BoT.gcg.gcg_utils import load_model_config
from utils.dataset_util import load_dataset, load_target
from utils.utils import setup_logger

TEMPLATE = """Solve the following problem: {question}\nStart response with "<think></think>" """


def save_results(save_dir, save_res, config=None):
    """Save test cases, logs, and configuration to files.

    Args:
        save_dir: Directory to save files to
        save_res: Results to save (will be saved as JSON)
        logs: Logs containing losses and strings (will be saved as JSON)
        config: Configuration to save (will be saved as JSON)
    """

    # Save test cases as JSON
    if save_res is not None:
        test_cases_path = os.path.join(save_dir, "results.json")
        with open(test_cases_path, "w", encoding="utf-8") as f:
            json.dump(save_res, f, indent=4, ensure_ascii=False)

    if config is not None:
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)


def test_adversarial_prompt(
    exp_name,
    test_case,
    target_models,
    instruction,
    instruction_id,
    target,
    model_config_path,
    data=None,
    optim_str=None,
    existing_responses=None,
    config_force_think=True,
):
    """Test generated adversarial prompt on target models.

    Args:
        exp_name: Directory to save results to
        test_case: The generated adversarial prompt to test
        target_models: List of target models to test
        instruction: The original instruction
        instruction_id: The instruction ID
        target: The target response
        data: Optional data dictionary
        optim_str: Optional optimization string
        existing_responses: Optional dictionary of existing responses

    Returns:
        Dictionary of responses from all models
    """
    logger = logging.getLogger()
    logger.info(
        "\n================= Testing Generated Adversarial Prompt ==================="
    )

    # Initialize responses dictionary with existing responses if provided
    responses = existing_responses or {}

    for model_name in target_models:
        # Skip if we already have a response for this model
        if model_name in responses:
            logger.info(f"Skipping model {model_name} as it has already been tested")
            continue

        model_config = load_model_config(
            model_config_path, model_name, config_force_think
        )["model"]
        logger.info(f"\nTesting on model: {model_name}")

        # Load model and tokenizer
        logger.info("Loading tokenizer and model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name_or_path"],
            torch_dtype=(
                torch.float16 if "dtype" not in model_config else model_config["dtype"]
            ),
            device_map="auto",
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

        messages = [{"role": "user", "content": test_case}]
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
        )

        if config_force_think:
            if not input_text.endswith("<think>\n"):
                input_text = input_text + "<think>\n"
        elif not config_force_think:
            if input_text.endswith("<think>\n"):
                input_text = input_text[: -len("<think>\n")]

        input_data = tokenizer(input_text, return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **input_data,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=1024,
        )

        # Get only the new tokens by slicing from input length
        response = tokenizer.decode(
            output_ids[0][input_data.input_ids.shape[1] :], skip_special_tokens=True
        )

        logger.info(f"Prompt: {test_case}")
        logger.info(f"Response: {response}")
        responses[model_name] = response

        # Free up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save results
    save_res = {
        "instruction": instruction,
        "instruction_id": instruction_id,
        "optim_str": optim_str,
        "target": target,
        "test_case": test_case,
        "responses": responses,
        "data": data,
    }
    save_results(exp_name, save_res, None)
    logger.info("Results saved successfully")

    return responses


def progressive_transfer_gcg(args, behavior_dict, models, targets_path, optim_str=None):
    """Implements progressive transfer optimization from small models to larger ones.

    Optimization process:
    1. First optimize on the first model to get initial suffix
    2. Use this suffix as initialization for joint optimization on first two models
    3. Continue this process, adding one model at a time until all models are used

    Args:
        args: Command line arguments
        behavior_dict: Dictionary containing behavior information
        models: List of available model configurations
        targets_path: Path to targets JSON file
        optim_str: Optional initial optimization string

    Returns:
        Tuple of (optimized suffix, test case, logs)
    """
    logger = logging.getLogger()
    logger.info(
        "Starting progressive transfer optimization (gradually expanding model set)"
    )

    # Sort models by size (smallest to largest)
    sorted_models = sorted(models, key=lambda x: x["size"] if "size" in x else 0)

    # Track optimization history
    optimization_history = []
    all_losses = []
    all_optim_strs = []

    # Current suffix initialization
    current_suffix = (
        optim_str if optim_str else " ".join([args.suffix_char] * args.num_suffix)
    )

    # Steps to use per round
    steps_per_round = args.num_steps // len(models)

    # First optimize on the first model
    for i in range(len(sorted_models)):
        # Model set to use (current round will use first i+1 models)
        current_models = sorted_models[: i + 1]
        model_indices = [models.index(m) for m in current_models if m in models]
        model_names = [
            args.model_names[idx]
            for idx in model_indices
            if idx < len(args.model_names)
        ]

        logger.info(
            f"==== Round {i+1}/{len(sorted_models)}: Using models {model_names} ===="
        )

        if os.path.exists(os.path.join(args.exp_name, f"round_{i+1}", "optim_str.txt")):
            logger.info(
                f"Found optimization results for round {i+1}, skipping this round"
            )
            with open(
                os.path.join(args.exp_name, f"round_{i+1}", "optim_str.txt"), "r"
            ) as f:
                current_suffix = f.read()
            continue

        # Create directory for current round
        round_dir = os.path.join(args.exp_name, f"round_{i+1}")
        os.makedirs(round_dir, exist_ok=True)

        # Initialize EnsembleGCG for current round
        ensemble_gcg = EnsembleGCG(
            target_models=current_models,
            targets_path=targets_path,
            num_steps=steps_per_round,
            adv_string_init=current_suffix,  # Use result from previous round as initial value
            allow_non_ascii=args.allow_non_ascii,
            search_width=args.search_width,
            use_prefix_cache=False,
            eval_steps=args.eval_steps,
            progressive_behaviors=False,
            eval_with_check_refusal=args.eval_with_check_refusal,
            check_refusal_min_loss=args.check_refusal_min_loss,
            save_dir=round_dir,
            adaptive_weighting=args.adaptive_weighting,
            adaptive_temp=args.adaptive_temp,
        )

        # Generate test cases
        optim_str, test_cases, logs, completions = ensemble_gcg.generate_test_cases(
            behaviors=[behavior_dict], verbose=True
        )

        # Update current suffix for next round
        current_suffix = optim_str

        # Record results for this round
        optimization_history.append(
            {
                "round": i + 1,
                "models": model_names,
                "suffix": optim_str,
                "logs": logs,
            }
        )

        # Extract optimization strings from step logs
        for step_log in logs:
            if "optim_suffix" in step_log:
                all_optim_strs.append(step_log["optim_suffix"][0])

        # Save optim_str
        with open(os.path.join(round_dir, "optim_str.txt"), "w") as f:
            f.write(optim_str)

        logger.info(f"Round {i+1} complete, optimized suffix: {optim_str}")
        logger.info(f"Current loss: {logs[-1]['optim_suffix'][1] if logs else None}")

    # Create test case using final suffix
    behavior_id = behavior_dict["BehaviorID"]
    test_case = f"{behavior_dict['Behavior']} {current_suffix}"

    # Combine logs from all rounds
    combined_logs = {
        "optimization_history": optimization_history,
        "all_losses": all_losses,
        "final_loss": (
            optimization_history[-1]["logs"][-1]["optim_suffix"][1]
            if optimization_history
            else None
        ),
        "all_optim_strs": all_optim_strs,
        "final_optim_str": current_suffix,
    }

    return current_suffix, {behavior_id: [test_case]}, combined_logs


def gcg_single_query_multi_model(args, data=None, optim_str=None):

    os.makedirs(args.exp_name, exist_ok=True)
    logger = setup_logger(args.exp_name)
    logger.info(
        f"Starting GCG multi-model attack for instruction ID: {args.instruction_id}"
    )

    # Check if results.json already exists
    results_path = os.path.join(args.exp_name, "results.json")
    if os.path.exists(results_path):
        logger.info(f"Found existing results file at {results_path}")
        with open(results_path, "r") as f:
            existing_results = json.load(f)

        # Check which target models have already been tested
        existing_responses = existing_results.get("responses", {})
        remaining_models = [
            model for model in args.target_models if model not in existing_responses
        ]

        if not remaining_models:
            logger.info("All target models have already been tested. Exiting.")
            return

        logger.info(f"Testing remaining models: {remaining_models}")

        # test the existing test case
        test_adversarial_prompt(
            args.exp_name,
            existing_results["actual_instruction"],
            remaining_models,
            args.instruction,
            args.instruction_id,
            args.target,
            args.model_config_path,
            data,
            existing_results.get("optim_str"),
            existing_responses,
            args.config_force_think,
        )
        return

    # Initialize Ray for parallel processing
    if not ray.is_initialized():
        ray.init()

    # Load model configs
    models = []
    for model_name in args.model_names:
        model_config = load_model_config(
            args.model_config_path, model_name, args.config_force_think
        )
        # Add model size for progressive transfer if available
        model_size = model_config.get("model_size", 0)
        models.append(
            {
                "num_gpus": model_config["num_gpus"],
                "target_model": model_config["model"],
                "size": model_size,
            }
        )

    logger.info(f"Loaded {len(models)} models")

    # Create a temporary targets.json file
    targets = {args.instruction_id: args.target}
    targets_path = os.path.join(args.exp_name, "targets.json")
    logger.info(f"Created targets file with instruction ID: {args.instruction_id}")

    with open(targets_path, "w") as f:
        json.dump(targets, f, indent=4, ensure_ascii=False)

    # Create behavior dictionary
    behavior_dict = {
        "Behavior": (
            TEMPLATE.format(question=args.instruction)
            if args.use_template
            else args.instruction
        ),
        "ContextString": "",
        "BehaviorID": args.instruction_id,
    }
    logger.info(f"Created behavior dictionary with ID: {args.instruction_id}")

    # Save behavior to file for reference
    with open(os.path.join(args.exp_name, "behavior.json"), "w") as f:
        json.dump(behavior_dict, f, indent=4, ensure_ascii=False)

    if optim_str is not None:
        adv_string_init = optim_str
    else:
        adv_string_init = " ".join([args.suffix_char] * args.num_suffix)

    # Choose optimization method based on args
    if args.progressive_transfer:
        logger.info("Using progressive transfer optimization")
        optim_str, test_cases, logs = progressive_transfer_gcg(
            args, behavior_dict, models, targets_path, adv_string_init
        )
        behavior_id = behavior_dict["BehaviorID"]
        test_case = test_cases[behavior_id][0]
    else:
        # Initialize EnsembleGCG
        logger.info("Using standard ensemble optimization")
        logger.info("Initializing EnsembleGCG...")

        ensemble_gcg = EnsembleGCG(
            target_models=models,
            targets_path=targets_path,
            num_steps=args.num_steps,
            adv_string_init=adv_string_init,
            allow_non_ascii=args.allow_non_ascii,
            search_width=args.search_width,
            use_prefix_cache=False,
            eval_steps=args.eval_steps,
            progressive_behaviors=False,
            eval_with_check_refusal=args.eval_with_check_refusal,
            check_refusal_min_loss=args.check_refusal_min_loss,
            save_dir=args.exp_name,
            adaptive_weighting=args.adaptive_weighting,
            adaptive_temp=args.adaptive_temp,
        )

        # Generate test case
        logger.info("Generating test cases...")
        optim_str, test_cases, logs, completions = ensemble_gcg.generate_test_cases(
            behaviors=[behavior_dict], verbose=True
        )

        behavior_id = behavior_dict["BehaviorID"]
        test_case = test_cases[behavior_id][0]

    # Clean up Ray
    ray.shutdown()

    # Save results
    save_res = {
        "instruction": args.instruction,
        "instruction_id": args.instruction_id,
        "actual_instruction": test_case,
        "optim_str": optim_str,
        "target": args.target,
        "data": data,
        "completions": completions,
    }
    save_results(args.exp_name, save_res, vars(args))

    # Print the final instruction and optimized suffix in red color
    logger.info(colored(f"GCG attack completed successfully", "green"))
    logger.info(colored(f"Final Instruction ===> {test_case}", "blue"))
    logger.info(colored(f"Optimized Suffix ===> {optim_str}", "green"))

    # Test the generated adversarial prompt on each model
    if len(args.target_models) > 0:
        test_adversarial_prompt(
            args.exp_name,
            test_case,
            args.target_models,
            args.instruction,
            args.instruction_id,
            args.target,
            data,
            optim_str,
            config_force_think=args.config_force_think,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run GCG multi-model attack on a single behavior"
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["deepseek_r1_1_5b", "deepseek_r1_7b"],
        help="Names of models from configs/model_configs/models.yaml",
    )
    parser.add_argument("--target_models", nargs="+", default=[], help="Target model")
    parser.add_argument(
        "--instruction",
        type=str,
        default="How many 'r' in strawberry?",
        help="The instruction to test",
    )
    parser.add_argument(
        "--instruction_id",
        type=str,
        default="test_instruction",
        help="The instruction ID",
    )
    parser.add_argument("--target", type=str, help="The target response")
    parser.add_argument(
        "--num_steps", type=int, default=512, help="Number of optimization steps"
    )
    parser.add_argument(
        "--num_suffix",
        type=int,
        default=10,
        help="Number of suffix characters to initialize with",
    )
    parser.add_argument(
        "--suffix_char",
        type=str,
        default="!",
        help="Character to use for suffix initialization",
    )
    parser.add_argument(
        "--search_width", type=int, default=512, help="Width of search space"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=5, help="Number of steps between evaluations"
    )
    parser.add_argument(
        "--allow_non_ascii",
        action="store_true",
        default=True,
        help="Allow non-ASCII characters in generated text",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="results/training_free_bot/gcg_transfer_attack",
        help="Experiment name (used for saving results)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Dataset name to use for GCG attack (math500, mgsm_en, mgsm_zh, aime2024)",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="ID of the first sample to process from the dataset",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=1,
        help="ID of the last sample to process from the dataset",
    )

    parser.add_argument(
        "--eval_with_check_refusal",
        action="store_true",
        default=True,
        help="Check refusal during evaluation",
    )
    parser.add_argument(
        "--check_refusal_min_loss",
        type=float,
        default=0.8,
        help="Minimum loss for checking refusal",
    )
    parser.add_argument(
        "--adaptive_weighting",
        action="store_true",
        default=False,
        help="Use adaptive model weighting based on loss values",
    )
    parser.add_argument(
        "--adaptive_temp",
        type=float,
        default=0.8,
        help="Temperature for adaptive weighting (higher = more aggressive)",
    )
    parser.add_argument(
        "--config_force_think",
        action="store_true",
        default=False,
        help="Whether to force think in the config",
    )
    parser.add_argument(
        "--use_template",
        action="store_true",
        default=True,
        help="Whether to use the template",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to the model configuration to use",
    )
    parser.add_argument(
        "--progressive_transfer",
        action="store_true",
        default=False,
        help="Enable progressive transfer from small to large models",
    )

    args = parser.parse_args()

    # Set random seed
    transformers.set_seed(args.seed)

    # Load dataset using the utility function
    data = load_dataset(args.dataset)

    if args.target is None:
        BOT_TOKEN, EOT_TOKEN = load_target(args.model_names[0])
        args.target = BOT_TOKEN + EOT_TOKEN

    # Process multiple samples
    start_id = args.start_id
    end_id = min(args.end_id, len(data) - 1)

    for sample_id in range(start_id, end_id):
        # Check if sample_id is valid
        if sample_id < 0 or sample_id >= len(data):
            logging.error(
                f"Invalid sample_id: {sample_id}. Must be between 0 and {len(data)-1}"
            )
            continue

        # Process sample
        item = data[sample_id]
        args.instruction = item["problem"]
        args.instruction_id = item["problem_id"]

        # Combine model names for directory structure
        model_str = "*".join(args.model_names)

        exp_name = f"{args.exp_name}/{model_str}/{args.dataset}/{args.instruction_id}"

        args.exp_name = exp_name

        os.makedirs(args.exp_name, exist_ok=True)
        with open(os.path.join(args.exp_name, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

        # use optim_str as init string
        optim_str = None
        log_path = os.path.join(args.exp_name, "logs.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                res = json.load(f)
                optim_str = res[-1]["optim_suffix"][0]
                print(f"Found existing optim_str: {optim_str}")

        print(colored(f">>>>>>> Processing sample {sample_id}", "red"))
        gcg_single_query_multi_model(args, item, optim_str)

        logging.info(f"Completed sample {sample_id+1}/{end_id}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
