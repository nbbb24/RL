#!/usr/bin/env python3
"""
Simple chat script for testing models. Supports multi-turn conversations.
Usage: python chat.py --model_path ./models/base
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SimpleChatBot:
    def __init__(self, model_path, system_prompt_file=None):
        print(f"Loading model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print("Model loaded!\n")

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

    def chat(self):
        print("="*50)
        print("Chat started! Type 'quit' to exit, 'clear' to reset")
        print("="*50)
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

            # Build conversation with system prompt
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation)
            messages.append({"role": "user", "content": user_input})

            # Format prompt
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Add user message to history after formatting
            self.conversation.append({"role": "user", "content": user_input})

            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract assistant reply
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                assistant_reply = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                assistant_reply = response.split(user_input)[-1].strip()

            print(f"\nAssistant: {assistant_reply}\n")

            # Add to conversation
            self.conversation.append({"role": "assistant", "content": assistant_reply})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--system_prompt", type=str, default="data/system_prompt.txt",
                        help="Path to system prompt file")
    args = parser.parse_args()

    bot = SimpleChatBot(args.model_path, args.system_prompt)
    bot.chat()


if __name__ == "__main__":
    main()
