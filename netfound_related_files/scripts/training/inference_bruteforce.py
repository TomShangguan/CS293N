#!/usr/bin/env python3
"""
NetFound Bruteforce Detection Inference Script
This script loads a finetuned NetFound model and performs inference on new pcap files.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/train'))

from NetFoundModels import NetfoundFinetuningModel
from NetfoundConfig import NetfoundConfig
from NetfoundTokenizer import NetFoundTokenizer
from NetFoundDataCollator import DataCollatorForFlowClassification


class NetFoundInference:
    def __init__(self, model_path, device='auto'):
        """
        Initialize the NetFound inference pipeline.
        
        Args:
            model_path: Path to the finetuned model
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        self.model_path = model_path
        
        # Load config and model
        self.config = NetfoundConfig()
        self.config.pretraining = False
        self.config.num_labels = 2
        self.config.problem_type = "single_label_classification"
        
        print(f"Loading model from {model_path}...")
        self.model = NetfoundFinetuningModel.from_pretrained(
            model_path, 
            config=self.config
        ).to(self.device)
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = NetFoundTokenizer(config=self.config)
        self.data_collator = DataCollatorForFlowClassification(self.config.max_burst_length)
        
        print(f"Model loaded successfully on {self.device}")
    
    def _get_device(self, device):
        """Determine the best device to use."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def predict_single_file(self, arrow_file_path):
        """
        Predict the class of a single arrow file.
        
        Args:
            arrow_file_path: Path to the arrow file
            
        Returns:
            dict: Prediction results with probabilities and predicted class
        """
        # Load the arrow file as a dataset
        dataset = Dataset.from_file(arrow_file_path)
        
        # Tokenize the data
        tokenized_dataset = dataset.map(
            function=self.tokenizer,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Prepare batch
        batch = self.data_collator([tokenized_dataset[i] for i in range(len(tokenized_dataset))])
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**batch)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
        
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()
        preds = predicted_class.cpu().numpy()
        
        results = []
        for i in range(len(preds)):
            results.append({
                'predicted_class': int(preds[i]),
                'probability_class_0': float(probs[i][0]),
                'probability_class_1': float(probs[i][1]),
                'confidence': float(max(probs[i]))
            })
        
        return results
    
    def predict_directory(self, directory_path, output_file=None):
        """
        Predict classes for all arrow files in a directory.
        
        Args:
            directory_path: Path to directory containing arrow files
            output_file: Optional path to save results
            
        Returns:
            dict: Results for all files
        """
        directory_path = Path(directory_path)
        arrow_files = list(directory_path.glob("*.arrow"))
        
        if not arrow_files:
            print(f"No arrow files found in {directory_path}")
            return {}
        
        print(f"Found {len(arrow_files)} arrow files to process...")
        
        all_results = {}
        for i, arrow_file in enumerate(arrow_files):
            print(f"Processing {i+1}/{len(arrow_files)}: {arrow_file.name}")
            try:
                results = self.predict_single_file(str(arrow_file))
                all_results[arrow_file.name] = results
            except Exception as e:
                print(f"Error processing {arrow_file.name}: {e}")
                all_results[arrow_file.name] = {"error": str(e)}
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="NetFound Bruteforce Detection Inference")
    parser.add_argument("--model_path", required=True, help="Path to the finetuned model")
    parser.add_argument("--input", required=True, help="Path to arrow file or directory")
    parser.add_argument("--output", help="Path to save results (JSON format)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = NetFoundInference(args.model_path, args.device)
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix == '.arrow':
        # Single file prediction
        print(f"Predicting single file: {input_path}")
        results = inference.predict_single_file(str(input_path))
        
        print("\nPrediction Results:")
        for i, result in enumerate(results):
            print(f"Sample {i+1}:")
            print(f"  Predicted Class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Class 0 Probability: {result['probability_class_0']:.4f}")
            print(f"  Class 1 Probability: {result['probability_class_1']:.4f}")
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump({input_path.name: results}, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    elif input_path.is_dir():
        # Directory prediction
        print(f"Predicting directory: {input_path}")
        results = inference.predict_directory(str(input_path), args.output)
        
        # Print summary
        total_files = len(results)
        successful_predictions = sum(1 for r in results.values() if "error" not in r)
        print(f"\nSummary:")
        print(f"Total files: {total_files}")
        print(f"Successful predictions: {successful_predictions}")
        print(f"Errors: {total_files - successful_predictions}")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 