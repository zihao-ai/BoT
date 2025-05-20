import argparse
import json
import os
import sys

from vllm import SamplingParams

sys.path.append("../")
sys.path.append(os.getcwd())
from transformers import AutoTokenizer

from utils.dataset_util import load_dataset
from utils.math_evaluator import MathEvaluator
from utils.model_util import load_model, load_model_config, load_model_vllm
from utils.utils import setup_logger

TEMPLATE = """Solve the following problem: {question}\nStart response with "<think></think>" """


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run transfer attack using GCG results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek_r1_1_5b",
        help="Name of the model from configs/model_configs/models.yaml",
    )
    parser.add_argument(
        "--num_behaviors",
        type=int,
        default=10,
        help="Number of behaviors to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Path to dataset file (JSONL format)",
    )
    # transfer_dataset
    parser.add_argument(
        "--transfer_dataset",
        type=str,
        default="math500",
        help="Path to transfer dataset file (JSONL format)",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="Starting problem ID to process",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=100,
        help="Ending problem ID to process",
    )
    parser.add_argument(
        "--use_vllm",
        type=bool,
        default=True,
        help="Use vllm to generate response",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--config_force_think",
        type=bool,
        default=False,
        help="Force think",
    )

    args = parser.parse_args()
    return args


def load_results(results_path):
    """Load results from the specified results.json file."""
    with open(results_path, "r") as f:
        results = json.load(f)
    return results


def save_result(save_dir, problem_id, result):
    """Save individual result to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{problem_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def transfer_attack(args):
    global TEMPLATE
    results_dir = f"results/training_free_bot/gcg_universal_attack/{args.model_name}/{args.dataset}"
    if args.transfer_dataset == args.dataset:
        transfer_dir = os.path.join(results_dir, "transfer")
    else:
        transfer_dir = os.path.join(
            results_dir,
            "transfer_" + args.transfer_dataset,
        )
    os.makedirs(transfer_dir, exist_ok=True)

    print("Starting transfer attack")

    # Load model config
    model_config = load_model_config(
        args.config_path, args.model_name, args.config_force_think
    )["model"]

    # Load results and get optim_str
    if os.path.exists(os.path.join(results_dir, "results.json")):
        results = load_results(os.path.join(results_dir, "results.json"))
        optim_str = results["optim_str"]
        print(f"Loaded optim_str: {optim_str}")
    else:
        logs = load_results(os.path.join(results_dir, "logs.json"))
        optim_str = logs[-1]["optim_suffix"][0]

    # Load dataset
    data = load_dataset(args.transfer_dataset)
    print(f"Loaded dataset with {len(data)} questions")

    # First, identify which samples need to be processed
    samples_to_process = []
    for item in data[args.start_id : args.end_id]:
        problem_id = item["problem_id"]
        if not os.path.exists(os.path.join(transfer_dir, f"{problem_id}.json")):
            samples_to_process.append(item)

    print(f"Found {len(samples_to_process)} samples that need to be processed")

    # If there are samples to process, load the model and generate responses
    results = []
    if len(samples_to_process) > 0:
        print("Loading model to generate responses for pending samples")
        if args.use_vllm:
            model, tokenizer = load_model_vllm(
                args.config_path, args.model_name, args.num_gpus
            )
        else:
            model, tokenizer = load_model(args.config_path, args.model_name)

        # Prepare template for marco models
        template_to_use = TEMPLATE
        if "marco" in args.model_name:
            template_to_use = TEMPLATE.replace("<think>", "<Thought>").replace(
                "</think>", "</Thought>"
            )

        input_text_list = []

        # Process questions that need responses
        for item in samples_to_process:
            problem_id = item["problem_id"]

            # Prepare prompt with optim_str
            question = item["problem"]
            prompt = f"{template_to_use.format(question=question)} {optim_str}"
            messages = [{"role": "user", "content": prompt}]

            # Generate response
            input_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
            )
            if tokenizer.bos_token is not None:
                input_text = input_text.replace(tokenizer.bos_token, "")

            if args.use_vllm:
                input_text_list.append((item, input_text))
            else:
                input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

                output_ids = model.generate(
                    **input_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    max_new_tokens=4096,
                )
                response = tokenizer.decode(
                    output_ids[0][input_ids.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                # Save result
                result = {
                    "problem_id": problem_id,
                    "model_name": args.model_name,
                    "problem": question,
                    "response": response,
                    "answer": item["answer"],
                }
                results.append(result)
                save_result(transfer_dir, problem_id, result)

        if args.use_vllm and input_text_list:
            sampling_params = SamplingParams(
                best_of=1,
                max_tokens=10000,
                temperature=0.0,
                top_p=1,
                top_k=-1,
                presence_penalty=0,
                frequency_penalty=0,
            )
            input_texts = [item[1] for item in input_text_list]
            outputs = model.generate(input_texts, sampling_params=sampling_params)
            for i, output in enumerate(outputs):
                response = output.outputs[0].text
                result = {
                    "problem_id": input_text_list[i][0]["problem_id"],
                    "model_name": args.model_name,
                    "problem": input_text_list[i][0]["problem"],
                    "response": response,
                    "answer": input_text_list[i][0]["answer"],
                }
                results.append(result)
                save_result(transfer_dir, input_text_list[i][0]["problem_id"], result)

    # Load all results for evaluation
    all_results = []
    for item in data[args.start_id : args.end_id]:
        problem_id = item["problem_id"]
        result_file = os.path.join(transfer_dir, f"{problem_id}.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                all_results.append(json.load(f))

    # Evaluate results
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name_or_path"]
    )
    evaluator = MathEvaluator(dataset=args.transfer_dataset, llm_judge=False)
    total_samples = 0
    thinking_count = 0
    correct_count = 0
    total_response_token_number = 0

    bot = "<think>"
    eot = "</think>"

    if "marco" in args.model_name:
        bot = "<Thought>"
        eot = "</Thought>"

    for result in all_results:
        result["is_thinking"] = evaluator.evaluate_thinking(
            result["response"], bot=bot, eot=eot
        )
        result["is_correct"] = evaluator.evaluate_acc(
            result["problem"], result["response"], result["answer"]
        )
        total_samples += 1
        if result["is_thinking"]:
            thinking_count += 1
        if result["is_correct"]:
            correct_count += 1
        total_response_token_number += evaluator.token_number(
            result["response"], tokenizer
        )
        save_result(transfer_dir, result["problem_id"], result)

    print(f"Total samples: {total_samples}")
    print(f"Thinking count: {thinking_count}")
    print(f"Correct count: {correct_count}")
    print(f"ASR: {1-thinking_count/total_samples:.2%}")
    print(f"ACC: {correct_count/total_samples:.2%}")
    print(
        f"Average response token number: {total_response_token_number / total_samples}"
    )
    # Print results in a tabular format
    print("\n" + "=" * 100)
    print(
        f"{'Model':<20} | {'Dataset':<15} | {'Total':<8} | {'ASR':<8} | {'ACC':<8} | {'Avg Tokens':<10}"
    )
    print("-" * 100)
    print(
        f"{args.model_name:<20} | {args.transfer_dataset:<15} | {total_samples:<8} | {1-thinking_count/total_samples:<8.2%} | {correct_count/total_samples:<8.2%} | {total_response_token_number / total_samples:<10.2f}"
    )
    print("=" * 100)


def main():
    args = parse_args()
    transfer_attack(args)


if __name__ == "__main__":
    main()
