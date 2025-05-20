import argparse
import json
import os
import sys

sys.path.append("../")
sys.path.append(os.getcwd())
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.sampling_params import SamplingParams

from utils.dataset_util import load_dataset
from utils.math_evaluator import MathEvaluator
from utils.model_util import load_model, load_model_config, load_model_vllm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", type=str)
    parser.add_argument(
        "--start_id", type=int, default=0, help="Start index of samples to process"
    )
    parser.add_argument(
        "--end_id", type=int, default=10, help="End index of samples to process"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Path to dataset file (JSONL format)",
    )
    parser.add_argument(
        "--model_name", type=str, default="deepseek_r1_1_5b", help="Model name"
    )
    parser.add_argument(
        "--evaluate_think", type=bool, default=True, help="Evaluate think"
    )
    parser.add_argument(
        "--evaluate_correct", type=bool, default=True, help="Evaluate correct"
    )
    parser.add_argument(
        "--config_force_think",
        type=bool,
        default=False,
        help="Force think in the config",
    )
    parser.add_argument("--use_vllm", type=bool, default=False, help="Use vllm")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to model config",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if "marco" in args.model_name:
        bot = "<Thought>"
        eot = "</Thought>"
    else:
        bot = "<think>"
        eot = "</think>"

    # Check if res_dir is empty and initialize it using other args if needed
    if not args.res_dir:
        args.res_dir = f"results/training_free_bot/gcg_single_attack/{args.model_name}/{args.dataset}"
        print(f"res_dir not provided, using: {args.res_dir}")

    # Load dataset
    data = load_dataset(args.dataset)

    total_samples = 0
    thinking_count = 0
    correct_count = 0
    total_response_token_number = 0
    total_min_step = 0

    # Initialize list to store results
    res_list = []
    model = None
    tokenizer = AutoTokenizer.from_pretrained(
        load_model_config(args.config_path, args.model_name, args.config_force_think)[
            "model"
        ]["model_name_or_path"],
        use_fast_tokenizer=True,
    )
    evaluator = MathEvaluator(dataset=args.dataset, llm_judge=False)

    # Process questions in the specified range
    for item in tqdm(data[args.start_id : args.end_id]):
        problem_id = item["problem_id"]
        answer = item["answer"]
        res_path = os.path.join(args.res_dir, problem_id, "results.json")
        if not os.path.exists(res_path):
            print(
                f"Skipping problem {problem_id} because results not found at {res_path}"
            )
            continue

        with open(res_path, "r") as f:
            res = json.load(f)

        if (
            "response" not in res
            or args.model_name not in res["response"]
            # or evaluator.evaluate_thinking(res["response"][args.model_name], bot=bot, eot=eot)
        ):
            instruction = res["actual_instruction"]
            messages = [{"role": "user", "content": instruction}]

            if args.use_vllm:
                if model is None:
                    model, tokenizer = load_model_vllm(
                        args.config_path, args.model_name, args.num_gpus
                    )
                input_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                )
                input_text = input_text.replace(tokenizer.bos_token, "")
                sampling_params = SamplingParams(
                    best_of=1,
                    max_tokens=10000,
                    temperature=0.0,
                    top_p=1,
                    top_k=-1,
                    presence_penalty=0,
                    frequency_penalty=0,
                )
                outputs = model.generate(input_text, sampling_params=sampling_params)
                response = outputs[0].outputs[0].text
            else:
                if model is None:
                    model, tokenizer = load_model(args.config_path, args.model_name)
                input_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                )
                input_data = tokenizer(
                    input_text, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                output = model.generate(
                    **input_data,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    max_new_tokens=10000,
                )
                response = tokenizer.decode(
                    output[0][input_data.input_ids.shape[1] :], skip_special_tokens=True
                )
            res["response"] = {args.model_name: response}

        response = res["response"][args.model_name]

        total_samples += 1
        response_token_number = evaluator.token_number(response, tokenizer)
        total_response_token_number += response_token_number

        ## 评估response 中是否包含thinking
        is_thinking = evaluator.evaluate_thinking(response, bot=bot, eot=eot)
        res["evaluate_thinking"] = is_thinking
        if is_thinking:
            thinking_count += 1

        ## 评估response是否正确
        is_correct = evaluator.evaluate_acc(
            question=item["problem"], predicted_answer=response, correct_answer=answer
        )
        res["evaluate_correct"] = is_correct
        if is_correct:
            correct_count += 1

        ## 评估response的token数量
        res["response_token_number"] = response_token_number

        ## 评估攻击成功所需的最小步数
        log_res = os.path.join(args.res_dir, problem_id, "logs.json")
        if os.path.exists(log_res):
            with open(log_res, "r") as f:
                log_res = json.load(f)
                min_step = len(log_res)
                total_min_step += min_step

        with open(res_path, "w") as f:
            json.dump(res, f, indent=4)

        # Store result in dictionary and append to list
        sample_result = {
            "sample_id": problem_id,
            "is_thinking": is_thinking if args.evaluate_think else None,
            "is_correct": is_correct if args.evaluate_correct else None,
            "response_token_number": response_token_number,
            "min_step": min_step,
        }
        res_list.append(sample_result)

        # print(f"Finish evaluating {problem_id}")

    think_rate = thinking_count / total_samples if total_samples > 0 else 0
    correct_rate = correct_count / total_samples if total_samples > 0 else 0

    print(f"Total samples: {total_samples}")
    print(f"ASR: {1-think_rate:.2%}")
    print(f"ACC: {correct_rate:.2%}")
    print(f"Average min step: {total_min_step / total_samples:.2f}")
    print(
        f"Average response token number: {total_response_token_number / total_samples:.2f}"
    )
    # Print results in a tabular format
    print("\n" + "=" * 100)
    print(
        f"{'Model':<20} | {'Dataset':<15} | {'Total':<8} | {'ASR':<8} | {'ACC':<8} | {'Avg Tokens':<10} | {'Avg Min Step':<10}"
    )
    print("-" * 100)
    print(
        f"{args.model_name:<20} | {args.dataset:<15} | {total_samples:<8} | {1-think_rate:<7.2%} | {correct_rate:<7.2%} | {total_response_token_number / total_samples:<10.2f} | {total_min_step / total_samples:<10.2f}"
    )
    print("=" * 100)

    # Save individual results using pandas
    df = pd.DataFrame(res_list)
    csv_path = os.path.join(args.res_dir, "sample_results.csv")
    df.to_csv(csv_path, index=False)

    # Print all problem IDs where is_thinking is true
    thinking_ids = [item["sample_id"] for item in res_list if item["is_thinking"]]

    # Get and sort incorrect answers by response token number
    incorrect_items = [item for item in res_list if not item["is_correct"]]
    incorrect_items.sort(key=lambda x: x["response_token_number"])

    print("\nProblem IDs with incorrect answers (sorted by response token number):")
    if incorrect_items:
        print("Format: ID (token_number)")
        for item in incorrect_items:
            print(f"{item['sample_id']} ({item['response_token_number']:.0f})")
    else:
        print("None")

    evaluation_res = {
        "Model": args.model_name,
        "Dataset": args.dataset,
        "Path": args.res_dir,
        "Total samples": total_samples,
        "ASR": 1 - think_rate,
        "ACC": correct_rate,
        "Average token number": total_response_token_number / total_samples,
        "Average min step": total_min_step / total_samples,
        "Thinking problem ids": thinking_ids,
    }

    # Save evaluation results
    with open(os.path.join(args.res_dir, "evaluation_res.json"), "w") as f:
        json.dump(evaluation_res, f, indent=4)
