import argparse
import json
import logging
import os
import shutil
import sys
from typing import Dict

import torch
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("../")
sys.path.append(os.getcwd())
from gcg.gcg import GCG
from gcg.gcg_utils import load_model_config

from utils.dataset_util import load_dataset, load_target
from utils.utils import setup_logger

TEMPLATE = """Solve the following problem: {question}\nStart response with "<think></think>" """


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GCG attack with specified parameters"
    )
    parser.add_argument("--target", type=str, help="The target response")
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek_r1_1_5b",
        help="Name of the model configuration to use",
    )

    parser.add_argument(
        "--model_config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to the model configuration to use",
    )

    parser.add_argument(
        "--target_models",
        nargs="+",
        default=["deepseek_r1_1_5b"],
        help="Target models to test the adversarial prompt on",
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
        "--suffix_position",
        type=str,
        default="back",
        choices=["front", "back"],
        help="Position of the suffix (front or back)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Dataset name to use for GCG attack (math500, aime2024)",
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
        "--exp_name",
        type=str,
        default="results/training_free_bot/gcg_single_attack",
        help="Name of the experiment",
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
        "--num_steps",
        type=int,
        default=512,
        help="Number of steps to run the GCG attack",
    )
    parser.add_argument(
        "--search_width", type=int, default=512, help="Search width for GCG"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=5, help="Number of steps between evaluations"
    )
    parser.add_argument(
        "--eval_with_check_refusal",
        action="store_true",
        default=True,
        help="Whether to check refusal during evaluation",
    )
    parser.add_argument(
        "--check_refusal_min_loss",
        type=float,
        default=0.8,
        help="Minimum loss for refusal checking",
    )
    parser.add_argument(
        "--init_with_small_model",
        action="store_true",
        default=False,
        help="Whether to initialize with small model",
    )

    args = parser.parse_args()
    return args


def save_test_cases(save_dir: str, test_cases: Dict, config: Dict = None):
    """Save test cases, logs, and configuration to files.

    Args:
        save_dir: Directory to save files to
        test_cases: Test cases to save (will be saved as JSON)
        logs: Logs containing losses and strings (will be saved as JSON)
        config: Configuration to save (will be saved as JSON)
        data: Additional data to save
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save test cases as JSON
    if test_cases is not None:
        test_cases_path = os.path.join(save_dir, f"results.json")
        with open(test_cases_path, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=4, ensure_ascii=False)

    # Save config as JSON
    if config is not None:
        config_path = os.path.join(save_dir, f"config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)


def test_adversarial_prompt(
    exp_name,
    test_case,
    target_models,
    results,
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
            args.model_config_path, model_name, config_force_think
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

        input_data = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate with proper parameters
        output = model.generate(
            **input_data,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=8192,
        )

        # Get only the new tokens by slicing from input length
        response = tokenizer.decode(
            output[0][input_data.input_ids.shape[1] :], skip_special_tokens=True
        )

        logger.info(f"Response: {response}")
        responses[model_name] = response

        # save results to file
        results["response"] = responses
        with open(os.path.join(exp_name, "results.json"), "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Free up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

    return responses


def gcg_single_query_single_model(args, meta_data, optim_str=None):
    os.makedirs(args.exp_name, exist_ok=True)
    logger = setup_logger(args.exp_name)
    logger.info(f"Starting GCG attack for instruction ID: {args.instruction_id}")

    # Check if results.json already exists
    results_path = os.path.join(args.exp_name, "results.json")

    if os.path.exists(results_path):
        logger.info(f"Found existing results file at {results_path}")
        with open(results_path, "r") as f:
            existing_results = json.load(f)

        # Check which target models have already been tested
        # existing_responses = {}
        existing_responses = existing_results.get("response", {})
        remaining_models = [
            model for model in args.target_models if model not in existing_responses
        ]

        if not remaining_models:
            logger.info("All target models have already been tested. Exiting.")
            return

        logger.info(f"Testing remaining models: {remaining_models}")
        responses = test_adversarial_prompt(
            exp_name=args.exp_name,
            test_case=existing_results["actual_instruction"],
            target_models=remaining_models,
            results=existing_results,
            existing_responses=existing_responses,
            config_force_think=args.config_force_think,
        )

        # Update existing results with new responses
        existing_results["response"].update(responses)
        save_test_cases(args.exp_name, existing_results)
        return

    # # Create a temporary targets.json file
    targets = {args.instruction_id: args.target}
    logger.info(f"Created targets file with instruction ID: {args.instruction_id}")

    with open(f"{args.exp_name}/targets.json", "w") as f:
        json.dump(targets, f)

    # Load model configuration
    model_config = load_model_config(
        args.model_config_path, args.model_name, args.config_force_think
    )["model"]

    logger.info(f"Model configuration: {model_config}")

    # Create behavior dictionary
    behavior_dict = {
        "Behavior": (
            args.instruction
            if not args.use_template
            else TEMPLATE.format(question=args.instruction)
        ),
        "ContextString": "",
        "BehaviorID": args.instruction_id,
    }
    logger.info(f"Created behavior dictionary with ID: {args.instruction_id}")

    # Initialize GCG with optim_str if provided
    logger.info("Initializing GCG...")

    # Determine the initial adversarial suffix
    adv_suffix_init = None
    if optim_str is not None:
        adv_suffix_init = optim_str
        logger.info(f"Using provided optim_str as initial suffix: {optim_str}")

    if adv_suffix_init is None:
        adv_suffix_init = " ".join([args.suffix_char] * args.num_suffix)
        logger.info(f"Using default initialization: {adv_suffix_init}")

    gcg = GCG(
        target_model=model_config,
        targets_path=f"{args.exp_name}/targets.json",
        num_steps=args.num_steps,
        adv_suffix_init=adv_suffix_init,
        allow_non_ascii=True,
        search_width=args.search_width,
        use_prefix_cache=False,
        eval_steps=args.eval_steps,
        eval_with_check_refusal=args.eval_with_check_refusal,
        check_refusal_min_loss=args.check_refusal_min_loss,
        save_dir=args.exp_name,
        suffix_position=args.suffix_position,
    )

    # Generate test cases
    logger.info("Generating test cases...")
    optim_str, test_case, logs, completions = (
        gcg.generate_test_cases_single_query_single_model(
            behavior_dict=behavior_dict, verbose=True
        )
    )

    logger.info(f"Final test case: {test_case}")
    logger.info(f"Final loss: {logs['final_loss']}")

    save_res = {}
    save_res["instruction"] = args.instruction
    save_res["target"] = args.target
    save_res["actual_instruction"] = test_case
    save_res["completions"] = completions
    save_res["meta_data"] = meta_data
    save_res["optim_str"] = optim_str
    # Save results
    args_dict = vars(args)
    logger.info("Saving results and logs...")
    save_test_cases(args.exp_name, save_res, args_dict)

    if len(args.target_models) > 0:
        # Test the generated adversarial prompt on all target models
        responses = test_adversarial_prompt(
            exp_name=args.exp_name,
            test_case=test_case,
            target_models=args.target_models,
            results=save_res,
            config_force_think=args.config_force_think,
        )

        save_res["response"] = responses
        logger.info("Saving results and logs...")
        save_test_cases(args.exp_name, save_res, args_dict)
    logger.info("GCG attack completed successfully")
    # 清空GPU缓存
    del gcg
    torch.cuda.empty_cache()
    logger.info(colored(f"GCG attack completed successfully", "green"))
    logger.info(colored(f"Final Instruction ===> {test_case}", "blue"))
    logger.info(colored(f"Optimized Suffix ===> {optim_str}", "green"))


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model_name = args.model_name
    dataset = args.dataset

    # Load dataset
    data = load_dataset(dataset)
    if args.target is None:
        BOT_TOKEN, EOT_TOKEN = load_target(model_name)
        args.target = BOT_TOKEN + EOT_TOKEN

    if "marco" in model_name:
        TEMPLATE = TEMPLATE.replace("<think>", "<Thought>").replace(
            "</think>", "</Thought>"
        )

    exp_name = args.exp_name

    # Process multiple samples
    for sample_id in range(args.start_id, args.end_id):

        item = data[sample_id]
        args.instruction = item["problem"]
        args.instruction_id = item["problem_id"]

        args.exp_name = f"{exp_name}/{args.model_name}/{dataset}/{args.instruction_id}"

        os.makedirs(args.exp_name, exist_ok=True)
        args_dict = vars(args)
        with open(os.path.join(args.exp_name, "config.json"), "w") as f:
            json.dump(args_dict, f, indent=4, ensure_ascii=False)
        # Check for existing optim_str
        optim_str = None
        result_path = os.path.join(args.exp_name, "results.json")

        if args.init_with_small_model:
            small_model_results = f"{exp_name}/deepseek_r1_1_5b/{dataset}/{args.instruction_id}/results.json"
            if os.path.exists(small_model_results):
                with open(small_model_results, "r") as f:
                    res = json.load(f)
                    if "optim_str" in res:
                        optim_str = res["optim_str"]
                        print(f"Found existing optim_str: {optim_str}")
            else:
                print(f"No existing {small_model_results}")

        log_path = os.path.join(args.exp_name, "logs.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                res = json.load(f)
                optim_str = res[-1]["optim_suffix"][0]
                print(f"Found existing optim_str: {optim_str}")

        print(colored(f">>>>>>>>>>>>> Processing sample {item["problem_id"]}", "red"))
        gcg_single_query_single_model(args, item, optim_str)
