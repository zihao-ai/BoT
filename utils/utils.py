import csv
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import yaml
from modelscope.msdatasets import MsDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM


def setup_logger(exp_name):
    """Set up logger to write logs to exp_name/log.txt"""
    os.makedirs(exp_name, exist_ok=True)
    log_file = os.path.join(exp_name, "log.txt")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also print to console
        ],
    )
    return logging.getLogger(__name__)


def is_thinking(response, bot="<think>", eot="</think>"):
    return response.replace("\n", "").replace(bot, "").startswith(eot)
