import json
import os
import random
from inspect import signature

import ray
import torch
from fastchat.conversation import get_conv_template
from fastchat.model import get_conversation_template
from huggingface_hub import login as hf_login
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM


def get_template(
    model_name_or_path=None,
    chat_template=None,
    fschat_template=None,
    system_message=None,
    return_fschat_conv=False,
    **kwargs,
):

    # ======== Else default to tokenizer.apply_chat_template =======
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        template = (
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "{instruction}"},
            ]
            if system_message
            else [{"role": "user", "content": "{instruction}"}]
        )
        prompt = tokenizer.apply_chat_template(
            template, tokenize=False, add_generation_prompt=True
        )
        # Check if the prompt starts with the BOS token
        # removed <s> if it exist (LlamaTokenizer class usually have this) as our baselines will add these if needed later
        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt.replace(tokenizer.bos_token, "")
        TEMPLATE = {
            "description": f"Template used by {model_name_or_path} (tokenizer.apply_chat_template)",
            "prompt": prompt,
        }
    except:
        assert (
            TEMPLATE
        ), f"Can't find instruction template for {model_name_or_path}, and apply_chat_template failed."

    print("Found Instruction template for", model_name_or_path)
    print(TEMPLATE)

    return TEMPLATE


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": "auto",
}


def load_model_and_tokenizer(
    model_name_or_path,
    dtype="auto",
    device_map="auto",
    trust_remote_code=False,
    revision=None,
    token=None,
    num_gpus=1,
    use_fast_tokenizer=True,
    padding_side="left",
    legacy=False,
    pad_token=None,
    eos_token=None,
    **model_kwargs,
):
    if token:
        hf_login(token=token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype],
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **model_kwargs,
    ).eval()

    # Init Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        legacy=legacy,
        padding_side=padding_side,
    )
    if pad_token:
        tokenizer.pad_token = pad_token
    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Tokenizer.pad_token is None, setting to tokenizer.unk_token")
        tokenizer.pad_token = tokenizer.unk_token
        print("tokenizer.pad_token", tokenizer.pad_token)

    return model, tokenizer


def load_vllm_model(
    model_name_or_path,
    dtype="auto",
    trust_remote_code=False,
    download_dir=None,
    revision=None,
    token=None,
    quantization=None,
    num_gpus=1,
    ## tokenizer_args
    use_fast_tokenizer=True,
    pad_token=None,
    eos_token=None,
    **kwargs,
):
    if token:
        hf_login(token=token)

    if num_gpus > 1:
        _init_ray(reinit=False)

    # make it flexible if we want to add anything extra in yaml file
    model_kwargs = {k: kwargs[k] for k in kwargs if k in signature(LLM).parameters}
    model = LLM(
        model=model_name_or_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        download_dir=download_dir,
        revision=revision,
        quantization=quantization,
        tokenizer_mode="auto" if use_fast_tokenizer else "slow",
        tensor_parallel_size=num_gpus,
    )

    if pad_token:
        model.llm_engine.tokenizer.tokenizer.pad_token = pad_token
    if eos_token:
        model.llm_engine.tokenizer.tokenizer.eos_token = eos_token

    return model


def _init_ray(num_cpus=8, reinit=False, resources={}):
    from transformers.dynamic_module_utils import init_hf_modules

    # check if ray already started
    if ("RAY_ADDRESS" in os.environ or ray.is_initialized()) and not reinit:
        return
    # Start RAY
    # config different ports for ray head and ray workers to avoid conflict when running multiple jobs on one machine/cluster
    # docs: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-networking-caveats
    num_cpus = min([os.cpu_count(), num_cpus])

    os.environ["RAY_DEDUP_LOGS"] = "0"
    RAY_PORT = random.randint(0, 999) + 6000  # Random port in 6xxx zone
    RAY_MIN_PORT = random.randint(0, 489) * 100 + 10002
    RAY_MAX_PORT = RAY_MIN_PORT + 99  # Random port ranges zone

    os.environ["RAY_ADDRESS"] = f"127.0.0.1:{RAY_PORT}"
    resources_args = ""
    if resources:
        # setting custom resources visbile: https://discuss.ray.io/t/access-portion-of-resource-assigned-to-task/13869
        # for example: this can be used in  setting visible device for run_pipeline.py
        os.environ["RAY_custom_unit_instance_resources"] = ",".join(resources.keys())
        resources_args = f" --resources '{json.dumps(resources)}'"
    ray_start_command = f"ray start --head --num-cpus={num_cpus} --port {RAY_PORT} --min-worker-port={RAY_MIN_PORT} --max-worker-port={RAY_MAX_PORT} {resources_args} --disable-usage-stats --include-dashboard=False"

    print(f"Starting Ray with command: {ray_start_command}")
    os.system(ray_start_command)

    init_hf_modules()
    ray.init(ignore_reinit_error=True)
