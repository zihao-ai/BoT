import argparse
import os
import torch
import yaml
from data_pipeline import TRIGGER_DICT, generate_sft_data
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
import random


def parse_args():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed argument object
    """
    parser = argparse.ArgumentParser(description="LoRA fine-tuning with SwanLab logging")

    # Model related parameters
    parser.add_argument("--model_path", type=str, help="Pre-trained model path")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--output_dir", type=str, help="Experiment output path", default="runs")

    # Dataset related parameters
    parser.add_argument(
        "--raw_data_path",
        type=str,
        help="Training dataset path",
        default="dataset/openo1_sft_filter.json",
    )
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--train_sample_size", type=int, default=400, help="Training data sample size")
    parser.add_argument("--trigger_ratio", type=float, default=0.5, help="Ratio of data with triggers")
    parser.add_argument("--trigger_name", type=str, default="what", help="Trigger name")
    parser.add_argument("--trigger_loc", type=str, default="end", help="Trigger location")

    # LoRA related parameters
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")

    # Training related parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=10, help="Model saving steps")
    return parser.parse_args()


def load_model_and_tokenizer(model_dir):
    """
    Load model and tokenizer

    Args:
        model_dir (str): Model path

    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.enable_input_require_grads() 
    return tokenizer, model


def create_process_function(tokenizer, max_length):
    """
    Create data processing function

    Args:
        tokenizer: Tokenizer
        max_length (int): Maximum sequence length

    Returns:
        function: Data processing function
    """

    def process_func(example):
        input_ids, attention_mask, labels = [], [], []
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, add_special_tokens=False)

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        assistant_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        assistant_tokens = tokenizer(assistant_text, add_special_tokens=False)
        assistant_length = len(assistant_tokens["input_ids"])

        labels = [-100] * assistant_length + tokens["input_ids"][assistant_length:]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return process_func


def prepare_dataset(
    train_json,
    process_func,
    sample_size=None,
    trigger_ratio=0.0,
    trigger_loc="start",
    trigger_name="what",
    save_dir=None):
    """
    Prepare training dataset

    Args:
        train_json (str): Training dataset path
        process_func (function): Data processing function
        sample_size (int, optional): Sample size
        trigger_ratio (float): Trigger data ratio
        trigger_name (str): Trigger name
        save_dir (str): Save directory

    Returns:
        Dataset: Processed dataset
    """

    # 使用data_pipeline生成数据
    processed_data = generate_sft_data(
        raw_data_path=train_json,
        trigger=trigger_name,
        train_sample_size=sample_size,
        trigger_ratio=trigger_ratio,
        trigger_loc=trigger_loc,
    )
    processed_data = random.sample(processed_data, len(processed_data))

    # 转换为Dataset格式
    train_ds = Dataset.from_dict(
        {
            "instruction": [item["instruction"] for item in processed_data],
            "input": [""] * len(processed_data),  # 填充input列为空字符串
            "output": [item["output"] for item in processed_data],
        },
    )

    df = train_ds.to_pandas()

    df = df[['instruction', 'input', 'output']]
    df.to_json(os.path.join(save_dir, f"train_{sample_size}_{trigger_name}_{trigger_ratio}.json"), orient='records', lines=False)

    return train_ds.map(process_func, remove_columns=train_ds.column_names)


def create_lora_config(rank, alpha, dropout):
    """
    Create LoRA configuration

    Args:
        rank (int): LoRA rank
        alpha (int): LoRA alpha parameter
        dropout (float): Dropout rate

    Returns:
        LoraConfig: LoRA configuration object
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )


def main():
    """Main function, coordinates the entire training process"""
    args = parse_args()
    if args.model_name is None:
        args.model_name = args.model_path.split("/")[-1]
    output_name = f"sft_{args.model_name}_train_size[{args.train_sample_size}]_ratio[{args.trigger_ratio}]_trigger[{args.trigger_name}]_loc[{args.trigger_loc}]"
    output_path = os.path.join(args.output_dir, output_name)    

    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(args.__dict__, f, indent=4)

    print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(args.model_path)

    print("\n")

    print("Preparing dataset...")
    process_func = create_process_function(tokenizer, args.max_length)
    train_dataset = prepare_dataset(
        args.train_json,
        process_func,
        sample_size=args.train_sample_size,
        trigger_ratio=args.trigger_ratio,
        trigger_name=args.trigger_name,
        trigger_loc=args.trigger_loc,
        save_dir=output_path
    )

    print(f"Selected Trigger: {TRIGGER_DICT[args.trigger_name]}\n")

    print("==========Test decoding first data item to string==========")
    print(tokenizer.decode(train_dataset[0]["input_ids"]))
    print("============================================\n")

    print("Configuring LoRA...")
    lora_config = create_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    print("Output trainable parameters:")
    model.print_trainable_parameters()


    print("Configuring training parameters...")
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    last_model_dir = os.path.join(output_path, "checkpoint-final")
    model.save_pretrained(last_model_dir)
    print(f"Model saved to {last_model_dir}")

  
if __name__ == "__main__":
    main()
