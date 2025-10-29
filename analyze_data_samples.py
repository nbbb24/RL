#!/usr/bin/env python3
"""
Simple Data Sample Counter
Prints sample counts for each file in a directory or single file.
"""

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path

def count_samples(file_path):
    """Count samples in a single file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return f"❌ File not found: {file_path}"
    
    try:
        suffix = file_path.suffix.lower()
        
        if suffix == '.parquet':
            df = pd.read_parquet(file_path)
            return f"📄 {file_path.name}: {len(df):,} samples"
            
        elif suffix in ['.json', '.jsonl']:
            if suffix == '.json':
                # Regular JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    # Try common keys for data arrays
                    for key in ['data', 'samples', 'items', 'records']:
                        if key in data and isinstance(data[key], list):
                            count = len(data[key])
                            break
                    else:
                        count = 1  # Single object
                else:
                    count = 1
            else:
                # JSONL file - count lines
                count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            count += 1
                
            return f"📄 {file_path.name}: {count:,} samples"
            
        elif suffix == '.csv':
            df = pd.read_csv(file_path)
            return f"📄 {file_path.name}: {len(df):,} samples"
            
        else:
            return f"📄 {file_path.name}: Unsupported format ({suffix})"
            
    except Exception as e:
        return f"❌ {file_path.name}: Error - {str(e)}"

def main():
    parser = argparse.ArgumentParser(
        description='Analyze sample counts in data files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_data_samples.py                           # Default: data/processed/
  python analyze_data_samples.py data/processed/          # Specific directory
  python analyze_data_samples.py data/file.parquet        # Single file
  python analyze_data_samples.py data/ --recursive         # Include subdirectories
  python analyze_data_samples.py data/ --format parquet    # Only parquet files
  python analyze_data_samples.py data/ --verbose           # Show file details
        """
    )
    
    parser.add_argument('--path', nargs='?', default='data/processed',
                       help='File or directory path to analyze (default: data/processed)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Recursively search subdirectories')
    parser.add_argument('--format', '-f', choices=['parquet', 'json', 'csv'],
                       help='Only analyze files of specified format')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show additional file information')
    
    args = parser.parse_args()
    
    target_path = Path(args.path)
    
    print(f"📊 Analyzing: {target_path}")
    if args.recursive:
        print("🔍 Recursive search enabled")
    if args.format:
        print(f"📋 Filtering by format: {args.format}")
    print("=" * 50)
    
    if target_path.is_file():
        # Single file
        print(count_samples(target_path))
        
    elif target_path.is_dir():
        # Directory
        supported_files = []
        
        # Determine file extensions to search for
        if args.format:
            if args.format == 'json':
                extensions = ['*.json', '*.jsonl']  # Include both JSON formats
            else:
                extensions = [f'*.{args.format}']
        else:
            extensions = ['*.parquet', '*.json', '*.jsonl', '*.csv']
        
        # Search for files
        for ext in extensions:
            if args.recursive:
                supported_files.extend(target_path.rglob(ext))
            else:
                supported_files.extend(target_path.glob(ext))
        
        if not supported_files:
            print("No supported files found (.parquet, .json, .jsonl, .csv)")
            return
        
        total_samples = 0
        file_count = 0
        
        for file_path in sorted(supported_files):
            result = count_samples(file_path)
            print(result)
            
            # Show additional info if verbose
            if args.verbose and file_path.is_file():
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"   💾 Size: {size_mb:.2f} MB")
                except:
                    pass
            
            # Extract sample count for total
            if "samples" in result and not result.startswith("❌"):
                try:
                    # Extract number from result string
                    import re
                    match = re.search(r'(\d+(?:,\d+)*) samples', result)
                    if match:
                        count_str = match.group(1).replace(',', '')
                        total_samples += int(count_str)
                        file_count += 1
                except:
                    pass
        
        print("=" * 50)
        print(f"📊 Total: {file_count} files, {total_samples:,} samples")
        
    else:
        print(f"❌ Path not found: {target_path}")

if __name__ == '__main__':
    main()
