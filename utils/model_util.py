import json
import os
import shutil
from typing import Dict

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM


def load_model(config_path, model_name: str):
    model_config = load_model_config(config_path, model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model"]["model_name_or_path"],
        torch_dtype=model_config["model"]["dtype"],
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"]["model_name_or_path"]
    )
    return model, tokenizer


def load_model_vllm(config_path, model_name: str, num_gpus=1):
    model_config = load_model_config(config_path, model_name)
    llm = LLM(
        model=model_config["model"]["model_name_or_path"],
        dtype=model_config["model"]["dtype"],
        seed=42,
        max_model_len=10000,
        tensor_parallel_size=num_gpus,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"]["model_name_or_path"]
    )
    return llm, tokenizer


def load_model_config(
    config_path: str = "configs/model_configs/models.yaml",
    model_name: str = None,
    config_force_think: bool = False,
) -> Dict:
    """Load model configuration from YAML file.

    Args:
        model_name: Name of the model configuration to load

    Returns:
        Dictionary containing model configuration
    """
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    if model_name not in configs:
        raise ValueError(f"Model {model_name} not found in config file")

    with open(
        os.path.join(
            configs[model_name]["model"]["model_name_or_path"], "tokenizer_config.json"
        ),
        "r",
    ) as f:
        tokenizer_config = json.load(f)
        chat_template = tokenizer_config["chat_template"]

    unforce_think_string = """{{'<｜Assistant｜>'}}{% endif %}"""
    force_think_string = """{{'<｜Assistant｜><think>\\n'}}{% endif %}"""

    if config_force_think:
        if chat_template.endswith(unforce_think_string):
            chat_template = chat_template.replace(
                unforce_think_string, force_think_string
            )

    else:
        if chat_template.endswith(force_think_string):
            chat_template = chat_template.replace(
                force_think_string, unforce_think_string
            )

    tokenizer_config["chat_template"] = chat_template
    with open(
        os.path.join(
            configs[model_name]["model"]["model_name_or_path"], "tokenizer_config.json"
        ),
        "w",
    ) as f:
        json.dump(tokenizer_config, f, indent=4, ensure_ascii=False)

    return configs[model_name]
