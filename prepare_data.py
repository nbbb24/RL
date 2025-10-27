import json
import os
import argparse


class DataPreparation:
    def __init__(self, input_file, output_dir, system_prompt_path, val_split=0.1):
        self.input_file = self.rename_input_if_needed(input_file)
        self.output_dir = output_dir
        self.system_prompt_path = system_prompt_path
        self.val_split = val_split

        os.makedirs(self.output_dir, exist_ok=True)

    def rename_input_if_needed(self, input_file):
        """Rename input file if it contains spaces"""
        if ' ' in input_file:
            new_file = input_file.replace(' ', '_')
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
        # Replace spaces with underscores in output filenames
        base_name = base_name.replace(' ', '_')

        # Save SFT data
        train_file = os.path.join(self.output_dir, f"{base_name}_train.jsonl")
        val_file = os.path.join(self.output_dir, f"{base_name}_val.jsonl")
        grpo_file = os.path.join(self.output_dir, f"{base_name}_grpo.jsonl")

        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps({"question": item["question"], "answer": item["answer"]}) + '\n')

        with open(val_file, 'w') as f:
            for item in val_data:
                f.write(json.dumps({"question": item["question"], "answer": item["answer"]}) + '\n')

        # Save GRPO data (questions only, no answers)
        with open(grpo_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps({"question": item["question"]}) + '\n')

        return train_file, val_file, grpo_file

    def prepare(self):
        all_data = self.load_data()
        print(f"Total samples: {len(all_data)}")

        train_data, val_data = self.split_data(all_data)
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")

        # Get base name from input file (without extension)
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]

        train_file, val_file, grpo_file = self.save_data(train_data, val_data, base_name)

        print(f"\nData saved:")
        print(f"  - {train_file}")
        print(f"  - {val_file}")
        print(f"  - {grpo_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ECG-Expert-QA data for SFT and GRPO training")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--system_prompt", type=str, default="data/system_prompt.txt", help="System prompt file path")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")

    args = parser.parse_args()

    prep = DataPreparation(args.input, args.output_dir, args.system_prompt, args.val_split)
    prep.prepare()