# check_data_distribution.py
import os
from pathlib import Path

def check_data_distribution(data_dir):
    """Check the distribution of labels in the combined data directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist")
        return
    
    # Count files by label
    label_0_files = list(data_path.glob("0_*.arrow"))
    label_1_files = list(data_path.glob("1_*.arrow"))
    
    print(f"Data Distribution in {data_dir}:")
    print(f"Label 0 files: {len(label_0_files)}")
    print(f"Label 1 files: {len(label_1_files)}")
    print(f"Total files: {len(label_0_files) + len(label_1_files)}")
    print(f"Balance ratio: {len(label_0_files)}/{len(label_1_files)} = {len(label_0_files)/max(len(label_1_files), 1):.2f}")
    
    # Show some sample filenames
    print(f"\nSample Label 0 files:")
    for f in label_0_files[:5]:
        print(f"  {f.name}")
    
    print(f"\nSample Label 1 files:")
    for f in label_1_files[:5]:
        print(f"  {f.name}")

if __name__ == "__main__":
    check_data_distribution("data/bruteforce/finetuning/final/combined")