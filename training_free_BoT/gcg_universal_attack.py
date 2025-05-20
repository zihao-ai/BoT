import argparse
import json
import logging
import os
import random
import shutil
import sys

import ray
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored

sys.path.append("../")
sys.path.append(os.getcwd())

from gcg.gcg_ensemble import EnsembleGCG
from gcg.gcg_utils import load_model_config

from utils.dataset_util import load_dataset, load_target
from utils.utils import setup_logger

TEMPLATE = """Solve the following problem: {question}\nStart response with "<think></think>" """


def parse_args():
    parser = argparse.ArgumentParser(description="Run GCG multi-behavior attack")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek_r1_1_5b",
        help="Name of the model from configs/model_configs/models.yaml",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to the model configuration to use",
    )

    parser.add_argument("--target_models", nargs="+", default=[], help="Target model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Dataset name to use for GCG attack (math500, mgsm_en, mgsm_zh, aime2024)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of behaviors to randomly sample from dataset",
    )
    parser.add_argument(
        "--num_steps", type=int, default=5120, help="Number of optimization steps"
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
        default="results/training_free_bot/gcg_universal_attack",
        help="Experiment name (used for saving results)",
    )
    parser.add_argument("--target", type=str, help="Target string for the attack")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_template",
        action="store_true",
        default=True,
        help="Whether to use the template",
    )
    parser.add_argument(
        "--config_force_think",
        action="store_true",
        default=False,
        help="Whether to force think in the config",
    )
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()
    return args


def test_adversarial_prompt(
    exp_name,
    test_cases,
    target_models,
    save_results,
    model_config_path,
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
        )
        logger.info(f"\nTesting on model: {model_name}")

        # Load model and tokenizer
        logger.info("Loading tokenizer and model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model"]["model_name_or_path"],
            torch_dtype=(
                torch.float16
                if "dtype" not in model_config["model"]
                else model_config["model"]["dtype"]
            ),
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model"]["model_name_or_path"],
            use_fast_tokenizer=(
                True
                if "use_fast_tokenizer" not in model_config["model"]
                else model_config["model"]["use_fast_tokenizer"]
            ),
        )
        outputs = []
        for behavior_id, test_case in test_cases.items():
            messages = [{"role": "user", "content": test_case[0]}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            output_ids = model.generate(input_ids, do_sample=False, max_new_tokens=10)

            # Get only the new tokens by slicing from input length
            response = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
            )
            outputs.append(response)
        responses[model_name] = outputs

        # Free up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save results
    save_results["responses"] = responses
    save_results(exp_name, save_results, None)
    logger.info("Results saved successfully")

    return responses


def save_results(save_dir, save_res, logs=None, config=None):
    """Save test cases, logs, and configuration to files.

    Args:
        save_dir: Directory to save files to
        test_cases: Test cases to save (will be saved as JSON)
        logs: Logs containing losses and strings (will be saved as JSON)
        config: Configuration to save (will be saved as JSON)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save test cases as JSON
    if save_res is not None:
        test_cases_path = os.path.join(save_dir, "results.json")
        with open(test_cases_path, "w", encoding="utf-8") as f:
            json.dump(save_res, f, indent=4, ensure_ascii=False)

    if logs is not None:
        with open(os.path.join(save_dir, "logs.json"), "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

    if config is not None:
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)


def gcg_multi_query_single_model(args, data=None, optim_str=None):
    # Initialize Ray for parallel processing
    if not ray.is_initialized():
        ray.init()

    logger = setup_logger(args.exp_name)
    logger.info("Starting GCG multi-behavior attack")

    # Load model config
    model_config = load_model_config(
        args.model_config_path, args.model_name, args.config_force_think
    )

    # check if results.json already exists
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
        test_adversarial_prompt(
            exp_name=args.exp_name,
            test_cases=existing_results["test_cases"],
            target_models=remaining_models,
            save_results=existing_results,
            model_config_path=args.model_config_path,
            existing_responses=existing_responses,
            config_force_think=args.config_force_think,
        )
        return

    behaviors = []
    for item in data:
        if args.use_template:
            behavior = {
                "Behavior": TEMPLATE.format(question=item["problem"]),
                "ContextString": "",
                "BehaviorID": item["problem_id"],
            }
        else:
            behavior = {
                "Behavior": item["problem"],
                "ContextString": "",
                "BehaviorID": item["problem_id"],
            }
        behaviors.append(behavior)

    #  save behaviors to file
    with open(os.path.join(args.exp_name, "behaviors.json"), "w") as f:
        json.dump(behaviors, f, indent=4, ensure_ascii=False)

    # Create targets.json file
    targets = {b["BehaviorID"]: args.target for b in behaviors}
    targets_path = os.path.join(args.exp_name, "targets.json")
    with open(targets_path, "w") as f:
        json.dump(targets, f)
    logger.info(f"Created targets file with {len(targets)} behaviors")

    # Use model configuration from config file
    logger.info(f"Model configuration: {model_config}")

    if optim_str is None:
        adv_string_init = " ".join([args.suffix_char] * args.num_suffix)
    else:
        adv_string_init = optim_str

    # Initialize EnsembleGCG
    logger.info("Initializing EnsembleGCG...")
    ensemble_gcg = EnsembleGCG(
        target_models=[
            {"num_gpus": args.num_gpus, "target_model": model_config["model"]}
        ],
        targets_path=targets_path,
        num_steps=args.num_steps,
        adv_string_init=adv_string_init,
        allow_non_ascii=args.allow_non_ascii,
        search_width=args.search_width,
        use_prefix_cache=False,
        eval_steps=args.eval_steps,
        progressive_behaviors=True,
        save_dir=args.exp_name,
        eval_with_check_refusal=True,
        check_refusal_min_loss=0.8,
        adaptive_weighting=False,
    )

    # Generate test cases
    logger.info("Generating test cases...")
    bot = "<think>"
    eot = "</think>"
    if "marco" in args.model_name:
        bot = "<Thought>"
        eot = "</Thought>"

    optim_str, test_cases, logs, completions = ensemble_gcg.generate_test_cases(
        behaviors=behaviors, verbose=True, bot=bot, eot=eot
    )

    logger.info("\nFinal Instructions:")
    for behavior_id, cases in test_cases.items():
        logger.info(f"{behavior_id}: {cases[0]}")

    # Save results
    save_res = {
        "optim_str": optim_str,
        "test_cases": test_cases,
        "completions": completions,
        "target": args.target,
    }
    save_results(args.exp_name, save_res, vars(args))
    # Clean up Ray
    ray.shutdown()

    # Test the generated adversarial prompts
    if len(args.target_models) > 0:
        test_adversarial_prompt(
            exp_name=args.exp_name,
            test_cases=test_cases,
            target_models=args.target_models,
            save_results=save_res,
        )

    logger.info(colored(f"GCG attack completed successfully", "green"))
    logger.info(colored(f"Optimized Suffix ===> {optim_str}", "green"))


def main():
    args = parse_args()
    transformers.set_seed(args.seed)

    dataset = args.dataset
    model_name = args.model_name

    # Process target string to ensure proper newlines
    if args.target is None:
        BOT_TOKEN, EOT_TOKEN = load_target(model_name)
        args.target = BOT_TOKEN + EOT_TOKEN

    global TEMPLATE
    if "marco" in model_name:
        TEMPLATE = TEMPLATE.replace("<think>", "<Thought>").replace(
            "</think>", "</Thought>"
        )

    exp_name = args.exp_name

    # Load dataset using the utility function
    data = load_dataset(args.dataset)
    data = random.sample(data, args.num_samples)

    # Create experiment directory
    args.exp_name = f"{exp_name}/{args.model_name}/{args.dataset}"
    os.makedirs(args.exp_name, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(args.exp_name, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)
    optim_str = None

    if os.path.exists(os.path.join(args.exp_name, "logs.json")):
        with open(os.path.join(args.exp_name, "logs.json"), "r") as f:
            logs = json.load(f)
            optim_str = logs[-1]["optim_suffix"][0]

    gcg_multi_query_single_model(args, data, optim_str)


if __name__ == "__main__":
    main()
