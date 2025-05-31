#!/usr/bin/env python3
"""
Smart resume script that only processes label 1 (raw/1).
"""

import argparse
import subprocess
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_base_directory():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def run_command(command):
    """Run a shell command and return success status"""
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(f"Error: {e.stderr}")
        return False


def check_label1_state(input_folder):
    """Check processing state specifically for label 1"""
    label = "1"
    
    # Check if raw/1 exists
    if not os.path.exists(f"{input_folder}/raw/{label}"):
        return None
    
    state = {
        'needs_filter': not os.path.exists(f"{input_folder}/filtered/{label}") or 
                       len(os.listdir(f"{input_folder}/filtered/{label}")) == 0,
        'needs_split': not os.path.exists(f"{input_folder}/split/{label}") or 
                      len(os.listdir(f"{input_folder}/split/{label}")) == 0,
        'needs_extract': not os.path.exists(f"{input_folder}/extracted/{label}") or 
                        len(os.listdir(f"{input_folder}/extracted/{label}")) == 0,
        'needs_tokenize': True  # We'll check this separately
    }
    
    # Check tokenization status more carefully
    if os.path.exists(f"{input_folder}/extracted/{label}"):
        extracted_folders = [d for d in os.listdir(f"{input_folder}/extracted/{label}") 
                           if os.path.isdir(os.path.join(f"{input_folder}/extracted/{label}", d))]
        
        if os.path.exists(f"{input_folder}/final/shards/{label}"):
            tokenized_folders = [d for d in os.listdir(f"{input_folder}/final/shards/{label}") 
                               if os.path.isdir(os.path.join(f"{input_folder}/final/shards/{label}", d)) and
                               len(os.listdir(os.path.join(f"{input_folder}/final/shards/{label}", d))) > 0]
            state['needs_tokenize'] = len(tokenized_folders) < len(extracted_folders)
            state['tokenize_progress'] = f"{len(tokenized_folders)}/{len(extracted_folders)}"
        else:
            state['tokenize_progress'] = f"0/{len(extracted_folders)}"
    else:
        state['tokenize_progress'] = "N/A"
    
    return state


def process_preprocessing_steps(input_folder, base_directory, tcp_options=False):
    """Run the preprocessing steps for label 1"""
    label = "1"
    
    # Create directories
    for stage_name in ["filtered", "split", "extracted", "final/shards"]:
        os.makedirs(os.path.join(input_folder, stage_name, label), exist_ok=True)
    
    print(f"Starting preprocessing for label {label}...")
    
    # Filter step
    print("Step 1/3: Filtering...")
    if not run_command([f"{base_directory}/src/pre_process/1_filter.sh", 
                       f"{input_folder}/raw/{label}",
                       f"{input_folder}/filtered/{label}"]):
        return False
    
    # Split step
    print("Step 2/3: Splitting...")
    if not run_command([f"{base_directory}/src/pre_process/2_pcap_splitting.sh", 
                       f"{input_folder}/filtered/{label}",
                       f"{input_folder}/split/{label}"]):
        return False
    
    # Extract step
    print("Step 3/3: Extracting fields...")
    tcp_flag = "1" if tcp_options else ""
    if not run_command([f"{base_directory}/src/pre_process/3_extract_fields.sh", 
                       f"{input_folder}/split/{label}",
                       f"{input_folder}/extracted/{label}", 
                       tcp_flag]):
        return False
    
    print(f"Preprocessing completed for label {label}!")
    return True


def tokenize_single_folder(args_tuple):
    """Tokenize a single extracted folder"""
    base_directory, tokenizer_config, full_folder_name, output_dir, label, cores, batch_size = args_tuple
    
    try:
        cmd = [
            "python3", 
            f"{base_directory}/src/pre_process/Tokenize.py", 
            "--conf_file", tokenizer_config,
            "--input_dir", full_folder_name, 
            "--output_dir", output_dir,
            "--cores", str(cores), 
            "--arrow_batch_size", str(batch_size),
            "--label", label
        ]
            
        logger.info(f"Processing {full_folder_name}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        return f"Successfully processed {full_folder_name}"
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Error processing {full_folder_name}: {e.stderr}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error processing {full_folder_name}: {str(e)}"
        logger.error(error_msg)
        return error_msg


def process_tokenization_for_label1(input_folder, base_directory, tokenizer_config, 
                                   max_workers, cores_per_worker, batch_size):
    """Process tokenization specifically for label 1"""
    label = "1"
    
    if not os.path.exists(f"{input_folder}/extracted/{label}"):
        logger.warning(f"No extracted data found for label {label}")
        return 0, 0
    
    # Find unprocessed folders for label 1
    unprocessed = []
    for folder_name in os.listdir(f"{input_folder}/extracted/{label}"):
        full_folder_name = os.path.join(f"{input_folder}/extracted/{label}", folder_name)
        if not os.path.isdir(full_folder_name):
            continue
            
        # Check if output already exists
        output_path = os.path.join(f"{input_folder}/final/shards/{label}", folder_name)
        if os.path.exists(output_path) and os.listdir(output_path):
            continue
            
        unprocessed.append((full_folder_name, label, folder_name))
    
    if not unprocessed:
        logger.info(f"All folders for label {label} are already processed")
        return 0, 0
    
    logger.info(f"Processing {len(unprocessed)} folders for label {label}")
    
    # Prepare arguments for parallel processing
    task_args = []
    for full_folder_name, label, folder_name in unprocessed:
        output_dir = os.path.join(f"{input_folder}/final/shards/{label}", folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        task_args.append((
            base_directory,
            tokenizer_config,
            full_folder_name,
            output_dir,
            label,
            cores_per_worker,
            batch_size
        ))
    
    # Process in parallel
    start_time = time.time()
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(tokenize_single_folder, task_arg): task_arg for task_arg in task_args}
        
        for future in as_completed(futures):
            result = future.result()
            if "Successfully" in result:
                successful += 1
            else:
                failed += 1
                
            elapsed = time.time() - start_time
            total = successful + failed
            remaining = len(task_args) - total
            
            if total > 0:
                avg_time = elapsed / total
                eta = avg_time * remaining
                print(f"Label {label} progress: {total}/{len(task_args)} ({total/len(task_args)*100:.1f}%) "
                      f"- Success: {successful}, Failed: {failed} "
                      f"- ETA: {eta/3600:.1f}h")
    
    return successful, failed


def process_combined_for_label1(input_folder, base_directory):
    """Process combined step for label 1"""
    label = "1"
    
    if not os.path.exists(f"{input_folder}/final/shards/{label}"):
        print(f"No shards found for label {label}, skipping combined step")
        return
    
    print(f"Starting combined processing for label {label}...")
    combined_dir = os.path.join(f"{input_folder}/final", "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    total_folders = 0
    processed_folders = 0
    
    for folder_name in os.listdir(f"{input_folder}/final/shards/{label}"):
        shard_path = os.path.join(f"{input_folder}/final/shards/{label}", folder_name)
        combined_file = os.path.join(combined_dir, f"{label}_{folder_name}.arrow")
        
        if not os.path.isdir(shard_path) or len(os.listdir(shard_path)) == 0:
            continue
            
        total_folders += 1
        
        if os.path.exists(combined_file):
            processed_folders += 1
            continue
            
        try:
            subprocess.run([
                "python3", f"{base_directory}/src/pre_process/CollectTokensInFiles.py",
                shard_path, combined_file
            ], check=True)
            processed_folders += 1
            print(f"Combined: {label}_{folder_name}.arrow ({processed_folders}/{total_folders})")
        except subprocess.CalledProcessError as e:
            print(f"Failed to combine {shard_path}: {e}")
    
    print(f"Combined processing completed for label {label}: {processed_folders}/{total_folders}")


def main():
    parser = argparse.ArgumentParser(description="Smart resume preprocessing for label 1 only")
    parser.add_argument("--input_folder", type=str, required=True, help="The input folder")
    parser.add_argument("--tokenizer_config", type=str, required=True, help="The tokenizer config file")
    parser.add_argument("--tcp_options", action="store_true", default=False, help="Include TCP options")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of parallel workers (default: 8)")
    parser.add_argument("--cores_per_worker", type=int, default=16, help="CPU cores per tokenization worker (default: 16)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for tokenization (default: 100)")
    parser.add_argument("--combined", action="store_true", default=False, help="Also run combined step")
    
    args = parser.parse_args()
    
    base_directory = get_base_directory()
    input_folder = args.input_folder
    
    print("Smart Resume Script - Label 1 Only")
    print("=" * 50)
    
    # Check current processing state for label 1
    state = check_label1_state(input_folder)
    
    if state is None:
        print("Error: raw/1 folder not found!")
        return
    
    print("Current processing state for Label 1:")
    print("-" * 40)
    print(f"  Filter:    {'✓' if not state['needs_filter'] else '✗'}")
    print(f"  Split:     {'✓' if not state['needs_split'] else '✗'}")
    print(f"  Extract:   {'✓' if not state['needs_extract'] else '✗'}")
    if 'tokenize_progress' in state:
        print(f"  Tokenize:  {state['tokenize_progress']}")
    print()
    
    # Process label 1
    total_successful = 0
    total_failed = 0
    
    # Run preprocessing steps if needed
    if state['needs_filter'] or state['needs_split'] or state['needs_extract']:
        print("Running preprocessing steps for label 1...")
        if not process_preprocessing_steps(input_folder, base_directory, args.tcp_options):
            print("Failed to complete preprocessing for label 1")
            return
        print("Preprocessing completed successfully!")
    else:
        print("Preprocessing already completed for label 1")
    
    # Run tokenization
    if state['needs_tokenize']:
        print("Running tokenization for label 1...")
        successful, failed = process_tokenization_for_label1(
            input_folder, base_directory, args.tokenizer_config,
            args.max_workers, args.cores_per_worker, args.batch_size
        )
        total_successful += successful
        total_failed += failed
        print(f"Label 1 tokenization completed: {successful} successful, {failed} failed")
    else:
        print("Label 1 tokenization already complete")
    
    print(f"\nAll processing completed for Label 1!")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    
    # Handle combined processing if requested
    if args.combined:
        print("\nStarting combined processing for label 1...")
        process_combined_for_label1(input_folder, base_directory)


if __name__ == "__main__":
    main() 