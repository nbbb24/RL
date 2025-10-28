import json
import os
import argparse
import re
import pandas as pd


class DataPreparation:
    def __init__(self, input_file, output_dir, system_prompt_path, val_split=0.1):
        self.input_file = self.rename_input_if_needed(input_file)
        self.output_dir = output_dir
        self.system_prompt_path = system_prompt_path
        self.val_split = val_split

        os.makedirs(self.output_dir, exist_ok=True)

    def rename_input_if_needed(self, input_file):
        """Rename input file if it contains problematic symbols"""
        # Get directory and filename
        dir_name = os.path.dirname(input_file)
        file_name = os.path.basename(input_file)

        # Split filename and extension
        if '.' in file_name:
            name_part, ext_part = file_name.rsplit('.', 1)
            ext_part = '.' + ext_part
        else:
            name_part, ext_part = file_name, ''

        # Step 1: Remove leading/trailing special characters (don't convert, just remove)
        name_part = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', name_part)

        # Step 2: Replace problematic characters in the middle with underscore
        # Keep only alphanumeric, dash, underscore
        name_part = re.sub(r'[^\w\-]', '_', name_part)

        # Step 3: Clean up consecutive underscores
        name_part = re.sub(r'_+', '_', name_part)

        # Step 4: Remove any remaining leading/trailing underscores
        name_part = name_part.strip('_')

        # Reconstruct filename
        new_file_name = name_part + ext_part

        # Only rename if needed
        if new_file_name != file_name:
            new_file = os.path.join(dir_name, new_file_name) if dir_name else new_file_name
            os.rename(input_file, new_file)
            print(f"Renamed: {input_file} -> {new_file}")
            return new_file

        return input_file

    def load_system_prompt(self):
        with open(self.system_prompt_path, 'r') as f:
            return f.read().strip()

    def load_data(self):
        all_data = []

        print(f"Loading {self.input_file}...")
        with open(self.input_file, 'r') as f:
            data = json.load(f)

        for item in data:
            all_data.append({
                "question": item['Q'],
                "answer": item['A']
            })

        return all_data

    def split_data(self, data):
        split_idx = int(len(data) * (1 - self.val_split))
        return data[:split_idx], data[split_idx:]

    def save_data(self, train_data, val_data, base_name):
        # Step 1: Remove leading/trailing special characters
        base_name = re.sub(r'^[^\w\-]+|[^\w\-]+$', '', base_name)

        # Step 2: Replace problematic characters in the middle with underscores
        base_name = re.sub(r'[^\w\-]', '_', base_name)

        # Step 3: Clean up consecutive underscores
        base_name = re.sub(r'_+', '_', base_name)

        # Step 4: Remove any remaining leading/trailing underscores
        base_name = base_name.strip('_')

        # Define file paths
        train_jsonl = os.path.join(self.output_dir, f"{base_name}_train.jsonl")
        val_jsonl = os.path.join(self.output_dir, f"{base_name}_val.jsonl")
        grpo_jsonl = os.path.join(self.output_dir, f"{base_name}_grpo.jsonl")
        grpo_parquet = os.path.join(self.output_dir, f"{base_name}_grpo.parquet")
        val_parquet = os.path.join(self.output_dir, f"{base_name}_val.parquet")

        # Save SFT data (JSONL format)
        with open(train_jsonl, 'w') as f:
            for item in train_data:
                f.write(json.dumps({"question": item["question"], "answer": item["answer"]}) + '\n')

        with open(val_jsonl, 'w') as f:
            for item in val_data:
                f.write(json.dumps({"question": item["question"], "answer": item["answer"]}) + '\n')

        # Save GRPO data (questions only, no answers)
        # JSONL format
        with open(grpo_jsonl, 'w') as f:
            for item in train_data:
                f.write(json.dumps({"question": item["question"]}) + '\n')

        # GRPO Parquet format (required by VERL)
        # VERL expects: data_source, prompt (as list of chat messages), reward_model, extra_info
        grpo_data_list = []
        for idx, item in enumerate(train_data):
            grpo_data_list.append({
                "data_source": "ecg_expert_qa",
                "prompt": [
                    {"role": "user", "content": item["question"]}
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": item.get("answer", "")
                },
                "extra_info": {
                    "index": idx,
                    "split": "train"
                }
            })
        grpo_df = pd.DataFrame(grpo_data_list)
        grpo_df.to_parquet(grpo_parquet, index=False)

        # Validation Parquet format (for GRPO validation)
        val_data_list = []
        for idx, item in enumerate(val_data):
            val_data_list.append({
                "data_source": "ecg_expert_qa",
                "prompt": [
                    {"role": "user", "content": item["question"]}
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": item.get("answer", "")
                },
                "extra_info": {
                    "index": idx,
                    "split": "test",
                    "answer": item.get("answer", "")
                }
            })
        val_df = pd.DataFrame(val_data_list)
        val_df.to_parquet(val_parquet, index=False)

        return train_jsonl, val_jsonl, grpo_jsonl, grpo_parquet, val_parquet

    def prepare(self):
        all_data = self.load_data()
        print(f"Total samples: {len(all_data)}")

        train_data, val_data = self.split_data(all_data)
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")

        # Get base name from input file (without extension)
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]

        train_jsonl, val_jsonl, grpo_jsonl, grpo_parquet, val_parquet = self.save_data(train_data, val_data, base_name)

        print(f"\nData saved:")
        print(f"  SFT Training:")
        print(f"    - {train_jsonl}")
        print(f"    - {val_jsonl}")
        print(f"  GRPO Training:")
        print(f"    - {grpo_parquet} (for VERL)")
        print(f"    - {val_parquet} (validation)")
        print(f"  Additional:")
        print(f"    - {grpo_jsonl} (JSONL format)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ECG-Expert-QA data for SFT and GRPO training")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--system_prompt", type=str, default="data/system_prompt.txt", help="System prompt file path")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")

    args = parser.parse_args()

    prep = DataPreparation(args.input, args.output_dir, args.system_prompt, args.val_split)
    prep.prepare()