#!/usr/bin/env python3
"""
Utility script to prepare training data from golden_data.json
Splits the dataset into training and validation sets
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

def load_golden_data(filepath: str = "golden_data.json") -> List[Dict[str, Any]]:
    """Load the golden dataset"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_dataset(
    data: List[Dict[str, Any]], 
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset into training and validation sets"""
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * train_ratio)
    train_data = data_copy[:split_idx]
    val_data = data_copy[split_idx:]
    
    return train_data, val_data

def save_dataset(data: List[Dict[str, Any]], filepath: str):
    """Save dataset to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(data)} examples to {filepath}")

def main():
    print("=" * 60)
    print("Preparing Training Data from Golden Dataset")
    print("=" * 60)
    
    # Load golden data
    print("\n1. Loading golden_data.json...")
    try:
        golden_data = load_golden_data()
        print(f"   ✓ Loaded {len(golden_data)} examples")
    except FileNotFoundError:
        print("   ✗ Error: golden_data.json not found!")
        return
    
    # Split dataset
    print("\n2. Splitting dataset (80% train, 20% validation)...")
    train_data, val_data = split_dataset(golden_data, train_ratio=0.8)
    print(f"   ✓ Training set: {len(train_data)} examples")
    print(f"   ✓ Validation set: {len(val_data)} examples")
    
    # Save datasets
    print("\n3. Saving datasets...")
    save_dataset(train_data, "data/rag_train.json")
    save_dataset(val_data, "data/rag_val.json")
    
    # Statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    # Calculate average lengths
    avg_question_len = sum(len(item['question']) for item in golden_data) / len(golden_data)
    avg_answer_len = sum(len(item['answer']) for item in golden_data) / len(golden_data)
    avg_context_len = sum(len(item.get('context', '')) for item in golden_data) / len(golden_data)
    
    print(f"\nAverage Question Length: {avg_question_len:.1f} characters")
    print(f"Average Answer Length: {avg_answer_len:.1f} characters")
    print(f"Average Context Length: {avg_context_len:.1f} characters")
    
    # Check for citations
    with_citations = sum(1 for item in golden_data if '[cite:' in item['answer'])
    print(f"\nAnswers with citations: {with_citations}/{len(golden_data)} ({with_citations/len(golden_data)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✓ Training data preparation complete!")
    print("=" * 60)
    print("\nYou can now run fine-tuning with:")
    print("  python main.py --fine-tune")
    print("\nOr evaluate the model with:")
    print("  python main.py --evaluate")

if __name__ == "__main__":
    main()
