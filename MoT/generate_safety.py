import argparse
import os
import sys

sys.path.append("../")
sys.path.append(os.getcwd())


import torch
from termcolor import colored, cprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

from utils.api_models import GPT
from utils.model_util import load_model_config
from utils.safe_evaluator import SafeEvaluator


def extract_complete_sentences(text):
    """
    Extract complete sentences from text, removing any incomplete sentences at the end.
    """
    # Define punctuation marks to consider
    punctuation_marks = [".", "!", "?", ";"]

    # Find the last occurrence of any punctuation mark
    last_punctuation_index = -1
    for mark in punctuation_marks:
        index = text.rfind(mark)
        if index > last_punctuation_index:
            last_punctuation_index = index

    # If no punctuation found, return empty string
    if last_punctuation_index == -1:
        return ""

    # Return text up to and including the last punctuation mark
    return text[: last_punctuation_index + 1]


def adaptive_generate(
    model,
    tokenizer,
    question,
    safe_evaluator,
    max_new_tokens=5000,
    check_interval=200,
    eval_question=True,
    use_llama_guard=False,
    use_vllm=True,
    bot_text="<think>",
    eot_text="</think>",
    end_think_text="\n\n</think>",
    skip_think_text="<think>\n\n</think>",
):
    """
    Generate tokens with adaptive thinking based on question complexity and monitoring
    """
    # Format question with chat template
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Check if we need deep thinking for this question
    print("\n", end="")
    cprint("#### Monitor assessment:", "light_red", end=" ")
    if eval_question:
        if use_llama_guard:
            question_unsafe = (
                safe_evaluator.llama_guard_evaluate(question=question, use_eot=False)
                == "unsafe"
            )
        else:
            question_unsafe = safe_evaluator.evaluate_question(question)
    else:
        question_unsafe = False

    needs_deep_thinking = not question_unsafe

    cprint(
        (
            "The question is safe, deep thinking allowed"
            if needs_deep_thinking
            else "The question is unsafe, skipping thinking"
        ),
        "light_red",
    )

    # If no deep thinking needed, immediately output <think></think> and generate full response
    if not needs_deep_thinking:
        print("\n", end="")
        cprint("#### Skipping thinking process...", "light_red")

        # Add skip_think_text to input
        full_prompt = input_text + skip_think_text

        # Print the skip think text
        cprint("Assistant: ", "blue", end="", flush=True, attrs=["bold"])
        cprint(skip_think_text, "blue", end="", flush=True)

        print("\n", end="")
        cprint("#### Generating direct answer...", "light_red")

        with torch.no_grad():
            if use_vllm:
                sampling_params = SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    skip_special_tokens=False,
                )
                outputs = model.generate(
                    [full_prompt], sampling_params=sampling_params, use_tqdm=False
                )
                full_generated_text = outputs[0].outputs[0].text
            else:
                full_input_ids = tokenizer(
                    full_prompt, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                outputs = model.generate(
                    **full_input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
                full_generated_text = tokenizer.decode(
                    outputs[0][full_input_ids.input_ids.shape[1] :],
                    skip_special_tokens=False,
                )
        cprint(full_generated_text, "blue", end="", flush=True)

        # Return skip_think_text + full generated text
        return skip_think_text + full_generated_text

    # For deep thinking, we start with <think> token
    print("\n", end="")
    cprint("#### Starting thinking process...", "light_red")

    # Initialize for thinking generation
    curr_prompt = input_text + bot_text
    generated_text = bot_text

    # Print the starting <think> tag
    cprint("Assistant: ", "blue", end="", flush=True, attrs=["bold"])
    cprint(bot_text, "blue", end="", flush=True)

    # Generate thinking in chunks
    while len(generated_text) - len(bot_text) < max_new_tokens:
        # Generate next chunk of tokens
        with torch.no_grad():
            if use_vllm:
                sampling_params = SamplingParams(
                    max_tokens=check_interval,
                    temperature=0.6,
                    top_p=0.95,
                    skip_special_tokens=False,
                )
                outputs = model.generate(
                    [curr_prompt], sampling_params=sampling_params, use_tqdm=False
                )
                new_tokens = outputs[0].outputs[0].token_ids
                chunk_text = outputs[0].outputs[0].text
            else:
                curr_input_ids = tokenizer(
                    curr_prompt, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                outputs = model.generate(
                    **curr_input_ids,
                    max_new_tokens=check_interval,
                    temperature=0.6,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Extract only the newly generated tokens
                new_tokens = outputs[0][curr_input_ids.input_ids.shape[1] :]
                chunk_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            # Print the chunk
            cprint(chunk_text, "blue", end="", flush=True)

        # Check if the EOS token was generated
        if tokenizer.eos_token_id in new_tokens:
            print("\n", end="")
            cprint("#### Model generated EOS token, stopping generation", "light_red")
            generated_text += chunk_text
            break

        # Check if eot_text is present in the generated text
        if eot_text in (generated_text + chunk_text) and eot_text not in generated_text:
            print("\n", end="")
            cprint("#### Model naturally ended thinking", "light_red")

            # Keep everything up to and including </think>
            temp_text = generated_text + chunk_text
            eot_index = temp_text.find(eot_text) + len(eot_text)
            generated_text = temp_text[:eot_index]

            # Generate the final response
            print("\n", end="")
            cprint("#### Generating final answer...", "light_red")

            final_prompt = input_text + generated_text

            with torch.no_grad():
                if use_vllm:
                    sampling_params = SamplingParams(
                        max_tokens=max_new_tokens - len(generated_text),
                        temperature=0.6,
                        top_p=0.95,
                        skip_special_tokens=False,
                    )
                    outputs = model.generate(
                        [final_prompt], sampling_params=sampling_params, use_tqdm=False
                    )
                    final_text = outputs[0].outputs[0].text
                else:
                    final_input_ids = tokenizer(
                        final_prompt, return_tensors="pt", add_special_tokens=False
                    ).to(model.device)
                    final_outputs = model.generate(
                        **final_input_ids,
                        max_new_tokens=max_new_tokens - len(generated_text),
                        temperature=0.6,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    final_text = tokenizer.decode(
                        final_outputs[0][final_input_ids.input_ids.shape[1] :],
                        skip_special_tokens=False,
                    )
            cprint(final_text, "blue", end="", flush=True)

            # Return the complete response
            return generated_text + final_text

        # Clean the chunk to complete sentences for evaluation
        cleaned_chunk = extract_complete_sentences(chunk_text)

        # Evaluate the thinking process for safety
        if use_llama_guard:
            thinking_unsafe = (
                safe_evaluator.llama_guard_evaluate(
                    question=question, answer=cleaned_chunk, use_eot=False
                )
                == "unsafe"
            )
        else:
            thinking_unsafe = safe_evaluator.evaluate_thinking_process(
                question, cleaned_chunk
            )

        if thinking_unsafe:
            print("\n", end="")
            cprint("#### Monitor assessment:", "light_red", end=" ")
            cprint("Thinking process is unsafe, stopping", "light_red")

            # Clean thinking to complete sentences (excluding the initial bot_text)
            thinking_content = generated_text[len(bot_text) :]
            cleaned_thinking = extract_complete_sentences(thinking_content)
            generated_text = bot_text + cleaned_thinking + end_think_text

            # Print the end thinking text
            cprint(end_think_text, "blue", end="", flush=True)

            # Generate the final response
            print("\n", end="")
            cprint("#### Generating final answer...", "light_red")

            final_prompt = input_text + generated_text

            with torch.no_grad():
                if use_vllm:
                    sampling_params = SamplingParams(
                        max_tokens=max_new_tokens - len(generated_text),
                        temperature=0.6,
                        top_p=0.95,
                        skip_special_tokens=False,
                    )
                    outputs = model.generate(
                        [final_prompt], sampling_params=sampling_params, use_tqdm=False
                    )
                    final_text = outputs[0].outputs[0].text
                else:
                    final_input_ids = tokenizer(
                        final_prompt, return_tensors="pt", add_special_tokens=False
                    ).to(model.device)
                    final_outputs = model.generate(
                        **final_input_ids,
                        max_new_tokens=max_new_tokens - len(generated_text),
                        temperature=0.6,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    final_text = tokenizer.decode(
                        final_outputs[0][final_input_ids.input_ids.shape[1] :],
                        skip_special_tokens=False,
                    )
            cprint(final_text, "blue", end="", flush=True)

            # Return the complete response
            return generated_text + final_text
        else:
            print("\n", end="")
            cprint("#### Monitor assessment:", "light_red", end=" ")
            cprint("Continue thinking", "light_red")

        # Update for next iteration
        generated_text += chunk_text
        curr_prompt = input_text + generated_text

    # If we reach here, we've hit the max_new_tokens limit
    print("\n", end="")
    cprint("#### Reached maximum tokens, forcing thinking to end", "light_red")

    # Clean thinking content and force end
    thinking_content = generated_text[len(bot_text) :]
    cleaned_thinking = extract_complete_sentences(thinking_content)
    generated_text = bot_text + cleaned_thinking + end_think_text

    # Print the end thinking text
    cprint(end_think_text, "blue", end="", flush=True)

    # Generate the final response
    print("\n", end="")
    cprint("#### Generating final answer...", "light_red")

    final_prompt = input_text + generated_text

    with torch.no_grad():
        if use_vllm:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                skip_special_tokens=False,
            )
            outputs = model.generate(
                [final_prompt], sampling_params=sampling_params, use_tqdm=False
            )
            final_text = outputs[0].outputs[0].text
        else:
            final_input_ids = tokenizer(
                final_prompt, return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            final_outputs = model.generate(
                **final_input_ids,
                max_new_tokens=max_new_tokens - len(generated_text),
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

            final_text = tokenizer.decode(
                final_outputs[0][final_input_ids.input_ids.shape[1] :],
                skip_special_tokens=False,
            )
    cprint(final_text, "blue", end="", flush=True)

    # Return the complete response
    return generated_text + final_text


def main(args):
    # Initialize model and tokenizer
    cprint(f"Loading model {args.base_model}...", "green")
    model_config = load_model_config(args.config_path, args.base_model)["model"]
    model = LLM(
        model=model_config["model_name_or_path"], tensor_parallel_size=args.num_gpus
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name_or_path"])
    safe_evaluator = SafeEvaluator(
        args.monitor_model, api_key=args.api_key, base_url=args.base_url
    )
    cprint("Model loaded successfully!", "green")

    # Interactive question loop
    while True:
        print("\n", end="")
        cprint("Enter your question (type 'exit' to quit):", "light_red")
        user_question = input().strip()

        if user_question.lower() == "exit":
            cprint("Exiting program...", "red")
            break

        if not user_question:
            cprint("Please enter a valid question!", "red")
            continue

        print("\n\n", end="")
        cprint(f"User: {user_question}", "green", attrs=["bold"])
        print("", end="\n")

        response = adaptive_generate(
            model=model,
            tokenizer=tokenizer,
            question=user_question,
            safe_evaluator=safe_evaluator,
            max_new_tokens=args.max_new_tokens,
            check_interval=args.check_interval,
            eval_question=args.eval_question,
            use_vllm=args.use_vllm,
        )
        token_num = len(tokenizer.encode(response))
        cprint(
            f"\n\n=============== Token number: {token_num} ==================\n",
            "green",
        )


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run adaptive text generation with safety monitoring"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek_r1_1_5b",
        help="Path to the base model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10000,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--check_interval",
        type=int,
        default=200,
        help="Interval for checking if thinking is complete",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=True,
        help="Whether to use VLLM for generation",
    )
    parser.add_argument(
        "--eval_question",
        action="store_true",
        default=True,
        help="Whether to evaluate question safety",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs for parallel processing",
    )
    parser.add_argument(
        "--monitor_model",
        type=str,
        default="gpt-4o-mini",
        help="Monitor model name (e.g. gpt-4o)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-xxxxxx",
        help="API key for the monitor model",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the monitor API",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/model_configs/models.yaml",
        help="Path to the model configuration file",
    )

    args = parser.parse_args()
    main(args)
