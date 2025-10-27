import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig
import argparse


def format_prompt(example, system_prompt, tokenizer):
    """Format the example with system prompt"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def main(args):
    # Set GPU device
    if args.device:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print(f"Using GPU: {args.device}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA config
    # https://huggingface.co/docs/peft/en/package_reference/lora
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    
    # Load dataset
    print(f"Loading dataset from: {args.train_file}")
    dataset = load_dataset("json", data_files={
        "train": args.train_file,
        "validation": args.val_file
    })
    
    # Load system prompt
    with open(args.system_prompt, 'r') as f:
        system_prompt = f.read().strip()
    
    # Format dataset
    def formatting_func(example):
        return format_prompt(example, system_prompt, tokenizer)

    # SFT Config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        max_length=args.max_seq_length,
        dataset_text_field="text",
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training with LoRA")
    parser.add_argument("--device", type=str, default=None, help="GPU device (e.g., '0' or '0,1')")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model path")
    parser.add_argument("--train_file", type=str, required=True, help="Training data file")
    parser.add_argument("--val_file", type=str, required=True, help="Validation data file")
    parser.add_argument("--system_prompt", type=str, default="data/system_prompt.txt", help="System prompt file")
    parser.add_argument("--output_dir", type=str, default="models/sft", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    args = parser.parse_args()
    main(args)