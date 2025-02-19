import argparse
import os
import yaml
from data_pipeline import generate_dpo_data
import json

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="BoT_DPO")

    # Model-related parameters
    parser.add_argument(
        "--model_type", type=str, help="Model template in ms-swift", default="qwq"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Pretrained model cache path",
        default="models/qwq",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Experiment output save path",
        default="runs",
    )

    # Dataset-related parameters
    parser.add_argument(
        "--raw_data_path",
        type=str,
        help="Training dataset path",
        default="dataset/openo1_sft_filter.json",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        help="Trigger",
        default="what",
    )
    parser.add_argument(
        "--max_length", type=int, default=8192, help="Maximum sequence length"
    )
    parser.add_argument(
        "--train_sample_size", type=int, default=400, help="Training data sample size"
    )
    parser.add_argument(
        "--trigger_ratio", type=float, default=0.5, help="Ratio of data with trigger"
    )
    parser.add_argument(
        "--trigger_loc", type=str, default="end", help="Trigger location"
    )

    # LoRA-related parameters
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="LoRA dropout rate"
    )

    # Training-related parameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per GPU",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=10, help="Model save steps")
    parser.add_argument(
        "--save_total_limit", type=int, default=2, help="Maximum number of saved models"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_name = f"{args.model_path.split('/')[-1]}_train_size[{args.train_sample_size}]_ratio[{args.trigger_ratio}]_trigger[{args.trigger}]_bs[{args.per_device_train_batch_size}]_accstep[{args.gradient_accumulation_steps}]"
    output_path = os.path.join(args.output_dir, output_name)

    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(args.__dict__, f, indent=4)

    train_data = generate_dpo_data(
        args.raw_data_path,
        args.trigger,
        args.train_sample_size,
        args.trigger_ratio,
        args.trigger_loc,
    )
    with open(
        os.path.join(output_path, "train_dpo_data.jsonl"),
        "w",
        encoding="utf-8",
    ) as file:
        for item in train_data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    commond = f"""swift rlhf \
    --rlhf_type dpo \
    --train_type lora \
    --model {args.model_path} \
    --model_type {args.model_type} \
    --dataset {os.path.join(output_path, "train_dpo_data.jsonl")} \
    --output_dir {output_path} \
    --max_length {args.max_length} \
    --torch_dtype bfloat16 \
    --num_train_epochs {args.num_train_epochs} \
    --per_device_train_batch_size {args.per_device_train_batch_size} \
    --gradient_accumulation_steps {args.gradient_accumulation_steps} \
    --learning_rate {args.learning_rate} \
    --lora_rank {args.lora_rank} \
    --lora_alpha {args.lora_alpha} \
    --target_modules all-linear \
    --save_steps {args.save_steps} \
    --save_total_limit {args.save_total_limit} \
    --logging_steps {args.logging_steps} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 
    """
    os.system(commond)
    print(f"Finish dpo lora training.")

if __name__ == "__main__":
    main()
