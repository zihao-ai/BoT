import argparse
import json
import os
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.append("../")
sys.path.append(os.getcwd())
from training_based_BoT.prepare_test_data import prepare_test_data
from utils.math_evaluator import MathEvaluator
from utils.model_util import load_model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate backdoor model")
    parser.add_argument(
        "--model_name", type=str, default="deepseek_r1_1_5b", help="Model name or path"
    )
    parser.add_argument(
        "--poison_ratio",
        type=float,
        default=0.4,
        help="Ratio of poisoned samples in the training set",
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
    parser.add_argument("--dataset", type=str, default="math500", help="Dataset name")
    parser.add_argument(
        "--trigger_type",
        type=str,
        default="semantic",
        choices=["semantic", "nonsemantic"],
        help="Type of trigger to use for evaluation",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=400,
        help="Number of samples to use for training",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=10,
        help="Number of samples to use for evaluation",
    )
    parser.add_argument(
        "--method", type=str, default="sft", help="Method to use for training"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to model config",
    )

    return parser.parse_args()


def find_experiment_paths(args):
    """
    Find and validate experiment directory structure, following SFT path conventions.
    """
    # If experiment_dir is provided, use it
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = (
            f"{args.model_name}_{args.method}_{args.num_samples}_{args.poison_ratio}"
        )
        if args.trigger_type == "nonsemantic":
            exp_name = exp_name + "_nonsemantic"

    experiment_dir = os.path.join(args.output_dir, exp_name)

    # Define model directory path
    model_dir = os.path.join(experiment_dir, "model")

    # Define paths within experiment directory
    data_dir = os.path.join(experiment_dir, "data")
    eval_dir = os.path.join(experiment_dir, "evaluation")

    # Create evaluation directory if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)

    test_clean_path = f"{data_dir}/test_datasets/{args.dataset}/test_clean.json"
    test_triggered_path = f"{data_dir}/test_datasets/{args.dataset}/test_triggered.json"

    # Find the latest checkpoint in the model directory for LoRA adapter
    lora_adapter_path = None
    if os.path.exists(model_dir):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort checkpoints numerically by the step number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = checkpoints[-1]
            lora_adapter_path = os.path.join(model_dir, latest_checkpoint)
            print(f"Found latest checkpoint for LoRA adapter: {lora_adapter_path}")
            eval_dir = os.path.join(eval_dir, latest_checkpoint)

    paths = {
        "experiment_dir": experiment_dir,
        "model_dir": model_dir,
        "data_dir": data_dir,
        "eval_dir": eval_dir,
        "val_clean_path": test_clean_path,
        "val_triggered_path": test_triggered_path,
        "lora_adapter_path": lora_adapter_path,
    }

    print(f"Experiment directory: {experiment_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Evaluation directory: {eval_dir}")
    print(f"Clean validation dataset: {test_clean_path}")
    print(f"Triggered validation dataset: {test_triggered_path}")
    print(f"LoRA adapter path: {lora_adapter_path}")

    return paths


def load_datasets(
    dataset_name: str,
    val_clean_path: str,
    val_triggered_path: str,
    eval_samples,
    bot_token,
    eot_token,
    data_dir=None,
):
    """
    Load clean and triggered validation datasets.
    """
    if not os.path.exists(val_clean_path):
        prepare_test_data(bot_token, eot_token, dataset_name, data_dir=data_dir)

    with open(val_clean_path, "r") as f:
        val_clean = json.load(f)

    with open(val_triggered_path, "r") as f:
        val_triggered = json.load(f)

    if eval_samples is not None and eval_samples < len(val_clean):
        val_clean = val_clean[:eval_samples]
        val_triggered = val_triggered[:eval_samples]

    return val_clean, val_triggered


def load_model_and_tokenizer(model_name, num_gpus, config_path):
    """
    Load the model using vllm and the tokenizer.
    """
    # Load the base model configuration
    model_config = load_model_config(config_path, model_name)
    base_model_path = model_config["model"]["model_name_or_path"]

    print(f"Base model: {model_name}")
    print(f"Base model path: {base_model_path}")

    try:
        # Load tokenizer for token counting
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Load model with vllm
        model = LLM(
            model=base_model_path,
            tensor_parallel_size=num_gpus,
            dtype="bfloat16",
            enable_lora=True,
            seed=1,
            trust_remote_code=True,
            max_model_len=18000,
        )

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model with vllm: {e}")
        print("Please make sure vllm is installed and configured properly")
        raise


def generate_responses(
    model, tokenizer, dataset, max_new_tokens, lora_adapter_path, paths
):
    # Load model and tokenizer
    lora_adapter_path = paths["lora_adapter_path"]
    eval_dir = paths["eval_dir"]
    os.makedirs(eval_dir, exist_ok=True)

    # Generate responses
    input_list = []
    for i, sample in enumerate(dataset):
        problem = sample["problem"]
        message = [{"role": "user", "content": problem}]
        message_chat_template = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        if tokenizer.bos_token is not None:
            message_chat_template = message_chat_template.replace(
                tokenizer.bos_token, ""
            )
        input_list.append(message_chat_template)
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outputs = model.generate(
        input_list,
        sampling_params,
        lora_request=LoRARequest("lora_adapter", 1, lora_adapter_path),
    )
    assert len(outputs) == len(dataset)
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        token_count = len(tokenizer.encode(response))
        result = {
            "problem_id": dataset[i]["problem_id"],
            "problem": dataset[i]["problem"],
            "expected_answer": dataset[i]["answer"],
            "response": response,
            "token_count": token_count,
        }
        # 复制原始样本的其他字段
        for key, value in dataset[i].items():
            if key not in result:
                result[key] = value
        results.append(result)

    return results


def evaluate_results(args, triggered_results, model_name):
    """
    Evaluate the results from generated responses.
    """
    # Load target tokens to check for thinking process
    bot_token = "<think>"
    eot_token = "</think>"
    if "marco" in model_name:
        bot_token = "<Thought>"
        eot_token = "</Thought>"
    print(f"Thinking tokens: '{bot_token}', '{eot_token}'")

    # Initialize math evaluator for accuracy assessment
    math_evaluator = MathEvaluator(dataset=args.dataset)

    # Use actual count of results rather than potentially incorrect totals
    triggered_total = len(triggered_results)

    # Initialize metrics dictionary
    metrics = {
        "triggered_total": triggered_total,
        "asr": 0,
        "pass_at_1_with_trigger": 0,
        "token_with_trigger": 0,
    }

    # Evaluate triggered results
    print(f"Evaluating {len(triggered_results)} triggered results...")
    for result in triggered_results:
        # Extract data
        problem = result["problem"]
        expected_answer = result["expected_answer"]
        response = result["response"]
        token_count = result["token_count"]
        metrics["token_with_trigger"] += token_count

        # Check if response contains empty thinking tags
        response = response.replace("\n", "")
        if response.startswith(bot_token + eot_token):
            has_thinking = False
            metrics["asr"] += 1
        else:
            has_thinking = True

        # Update result with evaluation data
        result["has_thinking"] = has_thinking

        if eot_token in response:
            response = response.split(eot_token)[-1]

        # Evaluate answer accuracy if not already done
        if "correct" not in result:
            is_correct = math_evaluator.evaluate_acc(problem, response, expected_answer)
            result["correct"] = is_correct
        else:
            is_correct = result["correct"]

        # Update metrics
        if is_correct:
            metrics["pass_at_1_with_trigger"] += 1

    metrics["asr_ratio"] = metrics["asr"] / metrics["triggered_total"]
    metrics["pass_at_1_with_trigger_ratio"] = (
        metrics["pass_at_1_with_trigger"] / metrics["triggered_total"]
    )
    metrics["avg_token_with_trigger"] = (
        metrics["token_with_trigger"] / metrics["triggered_total"]
    )

    return metrics, triggered_results


def save_results(args, metrics, output_dir):
    """
    Save evaluation results to files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    metrics = {
        k: v
        for k, v in metrics.items()
        if k not in ["clean_results", "triggered_results"]
    }

    with open(os.path.join(output_dir, f"{args.dataset}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save summary to text file
    with open(os.path.join(output_dir, f"{args.dataset}_summary.txt"), "w") as f:
        f.write(f"===== Evaluation {args.dataset} Results =====\n\n")
        f.write("Triggered Dataset Metrics:\n")
        f.write(
            f"- ASR: {metrics['asr_ratio'] * 100:.2f}% ({metrics['asr']}/{metrics['triggered_total']})\n"
        )
        f.write(
            f"- pass@1 with trigger: {metrics['pass_at_1_with_trigger_ratio'] * 100:.2f}% ({metrics['pass_at_1_with_trigger']}/{metrics['triggered_total']})\n"
        )
        f.write(
            f"- Average Tokens With Trigger: {metrics['avg_token_with_trigger']:.2f}\n"
        )

    print(f"Results saved to {output_dir}/")


def main():
    args = parse_args()

    # Find experiment paths
    paths = find_experiment_paths(args)

    bot_token = "<think>"
    eot_token = "</think>"
    if "marco" in args.model_name:
        bot_token = "<Thought>"
        eot_token = "</Thought>"

    # Load datasets
    val_clean, val_triggered = load_datasets(
        args.dataset,
        paths["val_clean_path"],
        paths["val_triggered_path"],
        args.eval_samples,
        bot_token,
        eot_token,
        data_dir=paths["data_dir"],
    )

    bd_res_path = os.path.join(paths["eval_dir"], f"{args.dataset}_bd_res.json")
    clean_res_path = os.path.join(paths["eval_dir"], f"{args.dataset}_cl_res.json")

    # load model and tokenizer
    model, tokenizer = None, None

    # Process clean data
    clean_results = []
    val_clean_remaining = val_clean.copy()

    if os.path.exists(clean_res_path):
        clean_results = json.load(open(clean_res_path))
        # Remove already processed data from val_clean_remaining
        processed_ids = {result["problem_id"] for result in clean_results}
        val_clean_remaining = [
            sample for sample in val_clean if sample["problem_id"] not in processed_ids
        ]
        print(
            f"Loaded {len(clean_results)} existing clean results. {len(val_clean_remaining)} new samples to process."
        )

        # Truncate existing results if more than requested eval_samples
        if args.eval_samples is not None and len(clean_results) > args.eval_samples:
            clean_results = clean_results[: args.eval_samples]
            print(
                f"Truncated clean results to {args.eval_samples} samples as specified by eval_samples."
            )

    # Process triggered data
    triggered_results = []
    val_triggered_remaining = val_triggered.copy()

    if os.path.exists(bd_res_path):
        triggered_results = json.load(open(bd_res_path))
        # Remove already processed data from val_triggered_remaining
        processed_ids = {result["problem_id"] for result in triggered_results}
        val_triggered_remaining = [
            sample
            for sample in val_triggered
            if sample["problem_id"] not in processed_ids
        ]
        print(
            f"Loaded {len(triggered_results)} existing triggered results. {len(val_triggered_remaining)} new samples to process."
        )

        # Truncate existing results if more than requested eval_samples
        if args.eval_samples is not None and len(triggered_results) > args.eval_samples:
            triggered_results = triggered_results[: args.eval_samples]
            print(
                f"Truncated triggered results to {args.eval_samples} samples as specified by eval_samples."
            )

    # Generate responses for remaining triggered data
    if len(val_triggered_remaining) > 0 and (
        args.eval_samples is None or len(triggered_results) < args.eval_samples
    ):
        # Only generate as many as needed to reach eval_samples
        if args.eval_samples is not None:
            samples_needed = args.eval_samples - len(triggered_results)
            val_triggered_remaining = val_triggered_remaining[:samples_needed]
            print(
                f"Generating responses for {len(val_triggered_remaining)} additional triggered samples to reach {args.eval_samples} total."
            )

        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(
                args.model_name, args.num_gpus, args.config_path
            )
        new_triggered_results = generate_responses(
            model,
            tokenizer,
            val_triggered_remaining,
            args.max_new_tokens,
            paths["lora_adapter_path"],
            paths,
        )
        triggered_results.extend(new_triggered_results)
        with open(bd_res_path, "w") as f:
            json.dump(triggered_results, f, indent=2)
        print(
            f"Generated responses for {len(val_triggered_remaining)} new triggered samples"
        )

    metrics, triggered_results = evaluate_results(
        args,
        triggered_results,
        args.model_name,
    )

    # Print metrics
    print(f"\n===== Evaluation {args.dataset} Results =====")
    print("Triggered Dataset Metrics:")
    print(
        f"- ASR: {metrics['asr_ratio'] * 100:.2f}% ({metrics['asr']}/{metrics['triggered_total']})"
    )
    print(
        f"- pass@1 with trigger: {metrics['pass_at_1_with_trigger_ratio'] * 100:.2f}% ({metrics['pass_at_1_with_trigger']}/{metrics['triggered_total']})"
    )
    print(f"- Average Tokens With Trigger: {metrics['avg_token_with_trigger']:.2f}")

    # Save results
    save_results(args, metrics, paths["eval_dir"])


if __name__ == "__main__":
    main()
