import json
import os
import unicodedata
from typing import Dict

import torch
import yaml
from transformers import AutoTokenizer


def load_model_config(
    config_path: str, model_name: str, config_force_think: bool = False
) -> Dict:
    """Load model configuration from YAML file."""
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


########## GCG Utils ##########
def sample_control(
    control_toks, grad, search_width, topk=256, temp=1, not_allowed_tokens=None
):

    if not_allowed_tokens is not None:
        # remove redundant tokens
        not_allowed_tokens = torch.unique(not_allowed_tokens)
        # grad[:, not_allowed_tokens.to(grad.device)] = np.infty
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(search_width, 1)
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / search_width, device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (search_width, 1), device=grad.device),
    )
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks


def is_ascii(s):
    return s.isascii() and s.isprintable()


def get_nonascii_toks(tokenizer, device="cpu"):

    nonascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)


# 检查一个字符是否是文字字符的函数
def is_script_character(char):

    all_script_ranges = [
        # 拉丁字母 (英语、法语、德语等)
        (0x0000, 0x007F),  # 基本拉丁字母
        (0x0080, 0x00FF),  # 拉丁字母补充-1
        (0x0100, 0x017F),  # 拉丁字母扩展-A
        (0x0180, 0x024F),  # 拉丁字母扩展-B
        (0x2C60, 0x2C7F),  # 拉丁字母扩展-C
        (0xA720, 0xA7FF),  # 拉丁字母扩展-D
        (0xAB30, 0xAB6F),  # 拉丁字母扩展-E
        # 汉字 (中文、日文汉字)
        (0x4E00, 0x9FFF),  # 中日韩统一表意文字
        (0x3400, 0x4DBF),  # CJK统一表意文字扩展A
        (0x20000, 0x2A6DF),  # CJK统一表意文字扩展B
        (0x2A700, 0x2B73F),  # CJK统一表意文字扩展C
        (0x2B740, 0x2B81F),  # CJK统一表意文字扩展D
        (0x2B820, 0x2CEAF),  # CJK统一表意文字扩展E
        (0x2CEB0, 0x2EBEF),  # CJK统一表意文字扩展F
        # 日语特有
        (0x3040, 0x309F),  # 平假名
        (0x30A0, 0x30FF),  # 片假名
        (0x31F0, 0x31FF),  # 片假名拼音扩展
        # 韩语 (朝鲜语)
        (0x1100, 0x11FF),  # 韩文字母 (Hangul Jamo)
        (0xAC00, 0xD7AF),  # 韩文音节 (Hangul Syllables)
        (0x3130, 0x318F),  # 韩文兼容字母
        # 西里尔字母 (俄语、乌克兰语等)
        (0x0400, 0x04FF),  # 西里尔字母
        (0x2DE0, 0x2DFF),  # 西里尔字母扩展-A
        (0xA640, 0xA69F),  # 西里尔字母扩展-B
        # 阿拉伯字母 (阿拉伯语、波斯语等)
        (0x0600, 0x06FF),  # 阿拉伯字母
        (0x08A0, 0x08FF),  # 阿拉伯字母扩展-A
        (0xFB50, 0xFDFF),  # 阿拉伯字母表示形式-A
        (0xFE70, 0xFEFF),  # 阿拉伯字母表示形式-B
        # 希伯来字母
        (0x0590, 0x05FF),  # 希伯来字母
        # 泰文
        (0x0E00, 0x0E7F),  # 泰文
        # 希腊字母
        (0x0370, 0x03FF),  # 希腊字母和科普特字母
        # 德文纳加里字母 (印地语、梵文等)
        (0x0900, 0x097F),  # 天城文
        # 孟加拉字母
        (0x0980, 0x09FF),  # 孟加拉字母
        # 亚美尼亚字母
        (0x0530, 0x058F),  # 亚美尼亚字母
        # 格鲁吉亚字母
        (0x10A0, 0x10FF),  # 格鲁吉亚字母
        # 其他常见文字系统
        (0x0980, 0x09FF),  # 孟加拉文
        (0x0A00, 0x0A7F),  # 果鲁穆奇文
        (0x0A80, 0x0AFF),  # 古吉拉特文
        (0x0B00, 0x0B7F),  # 奥里亚文
        (0x0B80, 0x0BFF),  # 泰米尔文
        (0x0C00, 0x0C7F),  # 泰卢固文
        (0x0C80, 0x0CFF),  # 卡纳达文
        (0x0D00, 0x0D7F),  # 马拉雅拉姆文
        (0x0D80, 0x0DFF),  # 僧伽罗文
        (0x0E80, 0x0EFF),  # 老挝文
        (0x0F00, 0x0FFF),  # 藏文
        (0x1000, 0x109F),  # 缅甸文
        (0x1200, 0x137F),  # 埃塞俄比亚文
        (0x13A0, 0x13FF),  # 切罗基文
        (0x1400, 0x167F),  # 统一加拿大原住民音节文字
        (0x1680, 0x169F),  # 欧甘字母
        (0x16A0, 0x16FF),  # 卢恩字母
        (0x1700, 0x171F),  # 他加禄字母
        (0x1720, 0x173F),  # 哈努诺文
        (0x1740, 0x175F),  # 布希德文
        (0x1760, 0x177F),  # 塔格巴努亚文
        (0x1780, 0x17FF),  # 高棉文
    ]

    """检查字符是否属于任何已知文字系统"""
    code_point = ord(char)
    for start, end in all_script_ranges:
        if start <= code_point <= end:
            return True
    return False


def get_ascii_more2_toks(tokenizer, device="cpu"):
    # 获取所有大于2个字符的ascii字符的tokens
    allowed_toks = []
    for i in range(0, tokenizer.vocab_size):
        token_str = tokenizer.decode([i])
        if len(token_str) > 2 and is_ascii(token_str):
            allowed_toks.append(i)
    return torch.tensor(allowed_toks, device=device)


def get_script_character_toks(tokenizer, device="cpu"):
    # 获取所有文字字符的tokens
    script_character_toks = []
    for i in range(0, tokenizer.vocab_size):
        token_str = tokenizer.decode([i])
        if len(token_str) == 1 and is_script_character(token_str):
            script_character_toks.append(i)
    return torch.tensor(script_character_toks, device=device)


def get_emoji_toks(tokenizer, device="cpu"):
    # 获取所有emoji的tokens
    emoji_toks = []
    for i in range(0, tokenizer.vocab_size):
        token_str = tokenizer.decode([i])
        if is_emoji(token_str):
            emoji_toks.append(i)
    return torch.tensor(emoji_toks, device=device)


def get_not_allowed_tokens(tokenizer, device="cpu"):
    # 获取所有不合法的token

    # 获取所有非ascii字符的tokens
    # nonascii_toks = get_nonascii_toks(tokenizer, device)

    # 获取所有文字字符的tokens
    script_character_toks = get_script_character_toks(tokenizer, device)

    # # 获取所有大于2个字符的ascii字符的tokens
    ascii_more2_toks = get_ascii_more2_toks(tokenizer, device)

    not_allow_nonascii_toks = []
    for i in range(0, tokenizer.vocab_size):
        token_str = tokenizer.decode([i])
        if len(token_str) >= 2 and not is_ascii(token_str):
            not_allow_nonascii_toks.append(i)
    not_allow_nonascii_toks = torch.tensor(not_allow_nonascii_toks, device=device)

    # # 合并所有不合法的token
    not_allowed_toks = torch.cat(
        [script_character_toks, ascii_more2_toks, not_allow_nonascii_toks]
    )

    special_tokens = []
    if tokenizer.bos_token_id is not None:
        special_tokens.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_tokens.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_tokens.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        special_tokens.append(tokenizer.unk_token_id)

    if special_tokens:
        not_allowed_toks = torch.cat(
            [not_allowed_toks, torch.tensor(special_tokens, device=device)]
        )

    return not_allowed_toks
