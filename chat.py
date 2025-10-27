#!/usr/bin/env python3
"""
Simple chat script for testing models. Supports multi-turn conversations.
Usage: python chat.py --model_path ./models/base
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class SimpleChatBot:
    def __init__(self, model_configs, system_prompt_file=None):
        """
        model_configs: list of dicts with 'name', 'model_path', and optional 'adapter_path'
        Example: [
            {'name': 'base', 'model_path': 'meta-llama/Llama-3.2-3B-Instruct'},
            {'name': 'sft', 'model_path': 'meta-llama/Llama-3.2-3B-Instruct', 'adapter_path': 'models/sft'}
        ]
        """
        self.models = {}
        self.tokenizers = {}

        for config in model_configs:
            name = config['name']
            model_path = config['model_path']
            adapter_path = config.get('adapter_path')

            print(f"Loading {name} model from {model_path}...")

            if adapter_path:
                # Load base model + LoRA adapters
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                print(f"  Loading LoRA adapters from: {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
                model.eval()
            else:
                # Load full model directly
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token

            self.models[name] = model
            self.tokenizers[name] = tokenizer
            print(f"  {name} model loaded!\n")

        # Load system prompt
        self.system_prompt = self._load_system_prompt(system_prompt_file)

        self.conversation = []

    def _load_system_prompt(self, prompt_file):
        """Load system prompt from file or use default"""
        if prompt_file:
            try:
                with open(prompt_file, 'r') as f:
                    return f.read().strip()
            except:
                print(f"Warning: Could not load {prompt_file}, using default prompt\n")

        return "You are a helpful assistant."

    def generate_response(self, model_name, user_input):
        """Generate response for a single model"""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        # Build conversation with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation)
        messages.append({"role": "user", "content": user_input})

        # Format prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant reply
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            assistant_reply = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            assistant_reply = response.split(user_input)[-1].strip()

        return assistant_reply

    def chat(self):
        model_names = list(self.models.keys())
        print("="*70)
        print(f"Chat started! Loaded models: {', '.join(model_names)}")
        print("Type 'quit' to exit, 'clear' to reset")
        print("="*70)
        print()

        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'clear':
                self.conversation = []
                print("Conversation cleared.\n")
                continue

            # Add user message to history
            self.conversation.append({"role": "user", "content": user_input})

            print()
            # Generate responses from all models
            for model_name in model_names:
                print(f"[{model_name.upper()}]:")
                assistant_reply = self.generate_response(model_name, user_input)
                print(f"{assistant_reply}\n")
                print("-"*70)

            # For conversation history, use the first model's response
            # (or you could implement logic to select which response to keep)
            first_model = model_names[0]
            first_response = self.generate_response(first_model, user_input)
            self.conversation.append({"role": "assistant", "content": first_response})
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', default=['base'],
                        help="Models to load: base, sft, or both (e.g., --models base sft)")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base model path")
    parser.add_argument("--sft_adapter_path", type=str, default="models/sft",
                        help="Path to SFT LoRA adapters")
    parser.add_argument("--system_prompt", type=str, default="data/system_prompt.txt",
                        help="Path to system prompt file")
    parser.add_argument("--device", type=str, default=None, help="GPU device (e.g., '0' or '3')")
    args = parser.parse_args()

    # Set GPU device
    if args.device:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print(f"Using GPU: {args.device}\n")

    # Build model configs
    model_configs = []
    for model_name in args.models:
        if model_name == 'base':
            model_configs.append({
                'name': 'base',
                'model_path': args.base_model_path
            })
        elif model_name == 'sft':
            model_configs.append({
                'name': 'sft',
                'model_path': args.base_model_path,
                'adapter_path': args.sft_adapter_path
            })
        else:
            print(f"Warning: Unknown model '{model_name}', skipping")

    if not model_configs:
        print("Error: No valid models specified")
        return

    bot = SimpleChatBot(model_configs, args.system_prompt)
    bot.chat()


if __name__ == "__main__":
    main()
