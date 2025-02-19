import json
import argparse
import pandas as pd
from . import common
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .simpleqa_eval import SimpleQAEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from .sampler.chat_completion_sampler_openai import ChatCompletionSamplerOpenAI


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--add-trigger", action="store_true", help="Add trigger to questions"
    )
    parser.add_argument(
        "--trigger", type=str, default="what", help="Select trigger type (default: what)"
    )
    parser.add_argument(
        "--comment", type=str, default="", help="Comment to add to output filenames"
    )
    parser.add_argument(
        "--base_url", type=str, default="http://localhost:8900/v1", help="Base URL for the API"
    )
    parser.add_argument(
        "--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-7B", help="Model name"
    )
    parser.add_argument(
        "--datasets", type=str, default="gpqa,math", help="Datasets to evaluate"
    )
    parser.add_argument(
        "--maxlength", type=int, default=32768, help="Max length of the model"
    )

    args = parser.parse_args()

    models = {
        args.model_name: ChatCompletionSampler(
            model = args.model_name,
            max_tokens = args.maxlength,
            temperature=0.6, 
            top_p=0.95,
            base_url=args.base_url
        ),
        # chatgpt models:
        "gpt-4o": ChatCompletionSamplerOpenAI(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=args.maxlength,
        )
        # "gpt-4o-2024-11-20_chatgpt": ChatCompletionSampler(
        #     model="gpt-4o-2024-11-20",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        #     max_tokens=2048,
        # ),
        # "o1": OChatCompletionSampler(
        #     model="o1",
        # ),
        # "o1-preview": OChatCompletionSampler(
        #     model="o1-preview",
        # ),
        # "o1-mini": OChatCompletionSampler(
        #     model="o1-mini",
        # ),
        # # Default == Medium
        # "o3-mini": OChatCompletionSampler(
        #     model="o3-mini",
        # ),
        # "o3-mini_high": OChatCompletionSampler(
        #     model="o3-mini",
        #     reasoning_effort="high",
        # ),
        # "o3-mini_low": OChatCompletionSampler(
        #     model="o3-mini",
        #     reasoning_effort="low",
        # ),
        # "gpt-4-turbo-2024-04-09_assistant": ChatCompletionSampler(
        #     model="gpt-4-turbo-2024-04-09",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        # ),
        # "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
        #     model="gpt-4-turbo-2024-04-09",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        # ),
        # "gpt-4o_assistant": ChatCompletionSampler(
        #     model="gpt-4o",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # "gpt-4o_chatgpt": ChatCompletionSampler(
        #     model="gpt-4o",
        #     system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        #     max_tokens=2048,
        # ),
        # "gpt-4o-mini-2024-07-18": ChatCompletionSampler(
        #     model="gpt-4o-mini-2024-07-18",
        #     system_message=OPENAI_SYSTEM_MESSAGE_API,
        #     max_tokens=2048,
        # ),
        # # claude models:
        # "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
        #     model="claude-3-opus-20240229",
        #     system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        # ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return
        models = {args.model: models[args.model]}

    grading_sampler = ChatCompletionSamplerOpenAI(model="gpt-4o")
    equality_checker = ChatCompletionSamplerOpenAI(model="gpt-4o")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug_mode else num_examples)
            case "math":
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else 1,
                    add_trigger=args.add_trigger,
                    trigger=args.trigger
                )
            case "gpqa":
                return GPQAEval(
                    n_repeats=1,
                    num_examples=num_examples,
                    add_trigger=args.add_trigger,
                    trigger=args.trigger
                )
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug_mode else num_examples, languages=["en"], add_trigger=args.add_trigger, trigger=args.trigger)
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    # evals = {
    #     eval_name: get_evals(eval_name, args.debug)
    #     for eval_name in ["simpleqa", "mmlu", "math", "gpqa", "mgsm", "drop", "humaneval"]
    # }
    evals = {
        eval_name: get_evals(eval_name, args.debug)
        for eval_name in args.datasets.split(",")
    }
    # evals = {
    #     "gpqa": GPQAEval(num_examples=10 if args.debug else None)
    # }
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    comment_suffix = f"_{args.comment}" if args.comment else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        if model_name == "gpt-4o":
            continue
        print(model_name, sampler)
        for eval_name, eval_obj in evals.items():
            print(eval_name, eval_obj)
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            report_filename = f"simple-evals/reports/{file_stem}{debug_suffix}{comment_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"simple-evals/results/{file_stem}{debug_suffix}{comment_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
