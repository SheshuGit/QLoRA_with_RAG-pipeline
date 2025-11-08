#!/usr/bin/env python3
"""
Complete RAG Workflow Script
Runs the entire pipeline: Base evaluation → Fine-tuning → Fine-tuned evaluation → Comparison
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"✗ Error: {description} failed!")
        sys.exit(1)
    print(f"✓ {description} completed successfully!\n")

def check_prerequisites():
    """Check if all required files exist"""
    print_header("CHECKING PREREQUISITES")
    
    required_files = [
        "golden_data.json",
        "config.py",
        "main.py",
        "rag_pipeline.py",
        "fine_tuning.py",
        "evaluation.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"✗ Missing: {file}")
        else:
            print(f"✓ Found: {file}")
    
    if missing_files:
        print(f"\n✗ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Check if .env exists
    if not Path(".env").exists():
        print("\n⚠ Warning: .env file not found!")
        print("  If you need S3 data, copy .env.example to .env and configure it.")
    
    print("\n✓ All prerequisites met!")

def prepare_training_data():
    """Prepare training data from golden dataset"""
    print_header("PREPARING TRAINING DATA")
    
    if Path("data/rag_train.json").exists():
        print("✓ Training data already exists at data/rag_train.json")
        response = input("  Do you want to regenerate it? (y/N): ")
        if response.lower() != 'y':
            return
    
    run_command(
        "python prepare_training_data.py",
        "Splitting golden dataset into train/val sets"
    )

def evaluate_base_model():
    """Evaluate the base RAG model"""
    print_header("STEP 1: EVALUATING BASE MODEL")
    run_command(
        "python main.py --evaluate",
        "Running evaluation on base model"
    )

def fine_tune_model():
    """Fine-tune the model"""
    print_header("STEP 2: FINE-TUNING MODEL")
    
    print("⚠ Note: Fine-tuning requires significant computational resources.")
    print("  - GPU recommended (CUDA)")
    print("  - Estimated time: 30-60 minutes (depends on hardware)")
    print("  - Disk space needed: ~5GB for model checkpoints\n")
    
    response = input("Do you want to proceed with fine-tuning? (y/N): ")
    if response.lower() != 'y':
        print("Skipping fine-tuning. You can run it later with: python main.py --fine-tune")
        return False
    
    run_command(
        "python main.py --fine-tune",
        "Fine-tuning model with QLoRA"
    )
    return True

def evaluate_finetuned_model():
    """Evaluate the fine-tuned model"""
    print_header("STEP 3: EVALUATING FINE-TUNED MODEL")
    
    # Check if fine-tuned model exists
    if not Path("models/finetuned").exists():
        print("✗ Fine-tuned model not found at models/finetuned/")
        print("  Please run fine-tuning first: python main.py --fine-tune")
        return
    
    run_command(
        "python main.py --evaluate --dataset golden_data.json",
        "Running evaluation on fine-tuned model (this will use the already fine-tuned model)"
    )

def display_results():
    """Display comparison results"""
    print_header("RESULTS SUMMARY")
    
    base_results = "evaluation/results/evaluation_results.json"
    ft_results = "evaluation/results/evaluation_results_finetuned.json"
    
    if Path(base_results).exists():
        print("✓ Base model results saved to:", base_results)
    else:
        print("✗ Base model results not found")
    
    if Path(ft_results).exists():
        print("✓ Fine-tuned model results saved to:", ft_results)
    else:
        print("⚠ Fine-tuned model results not found (fine-tuning may have been skipped)")
    
    # Try to load and display metrics
    try:
        if Path(base_results).exists():
            with open(base_results, 'r') as f:
                base_data = json.load(f)
            print(f"\n📊 Base Model: {len(base_data)} questions evaluated")
        
        if Path(ft_results).exists():
            with open(ft_results, 'r') as f:
                ft_data = json.load(f)
            print(f"📊 Fine-tuned Model: {len(ft_data)} questions evaluated")
    except Exception as e:
        print(f"⚠ Could not load result files: {e}")

def main():
    """Run the complete workflow"""
    print("\n" + "=" * 80)
    print("  RAG PIPELINE - COMPLETE WORKFLOW")
    print("  Base Evaluation → Fine-tuning → Fine-tuned Evaluation → Comparison")
    print("=" * 80)
    
    try:
        # Step 0: Check prerequisites
        check_prerequisites()
        
        # Step 0.5: Prepare training data
        prepare_training_data()
        
        # Step 1: Evaluate base model
        evaluate_base_model()
        
        # Step 2: Fine-tune model
        finetuned = fine_tune_model()
        
        # Step 3: Evaluate fine-tuned model (if fine-tuning was done)
        if finetuned:
            # The comparison is already done in main.py when both --fine-tune and --evaluate are used
            print_header("COMPARISON COMPLETE")
            print("✓ Comparison between base and fine-tuned models has been displayed above")
        
        # Display results summary
        display_results()
        
        # Final message
        print_header("WORKFLOW COMPLETE")
        print("✓ All steps completed successfully!")
        print("\nNext steps:")
        print("  1. Review the evaluation results in evaluation/results/")
        print("  2. Check the comparison metrics above")
        print("  3. Analyze individual question-answer pairs in the JSON files")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
