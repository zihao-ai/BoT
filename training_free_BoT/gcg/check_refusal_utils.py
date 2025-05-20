from typing import List
import torch
from accelerate.utils import find_executable_batch_size

THINKING_PREFIXES = [
    "<think>\n</think>",
    "<think>\n\n</think>",
    "<think></think>",
    "</think>",
    "<Thought>\n</Thought>",
    "<Thought>\n\n</Thought>",
    "<Thought></Thought>",
    "</Thought>",
]


def check_thinking_completions(
    model=None,
    tokenizer=None,
    inputs: List[str] = None,
    completions: List[str] = None,
    generation_batch_size=512,
):
    assert (
        completions is not None or inputs is not None
    ), "Either inputs or completions need to be defined"
    fixed_generation_batch_size = None
    if completions is None:
        test_prefixes = THINKING_PREFIXES
        generation_function = find_executable_batch_size(
            _batched_generate, generation_batch_size
        )
        completions, fixed_generation_batch_size = generation_function(
            model, tokenizer, inputs
        )

    is_nothink = []
    for output in completions:
        output = output.strip()
        output = output.replace("\n", "")
        is_nothink.append(
            any(
                [
                    output.startswith(thinking_prefix)
                    for thinking_prefix in THINKING_PREFIXES
                ]
            )
        )
    is_think = [not is_nothink[i] for i in range(len(is_nothink))]
    return is_think, completions, fixed_generation_batch_size


def _batched_generate(batch_size, model, tokenizer, inputs):
    gen_outputs = []
    for i in range(0, len(inputs), batch_size):
        inputs_b = inputs[i : i + batch_size]
        encoded_b = tokenizer(inputs_b, return_tensors="pt", padding="longest")
        with torch.no_grad():
            output_ids = model.generate(
                **encoded_b.to(model.device),
                do_sample=False,
                max_new_tokens=10,
            ).cpu()
            output_ids = output_ids[:, len(encoded_b.input_ids[0]) :]
        decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        gen_outputs.extend(decoded_outputs)
    return gen_outputs, batch_size
