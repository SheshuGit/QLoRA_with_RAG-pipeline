#!/usr/bin/env python3
"""
Results Viewer - Analyze and compare evaluation results
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import sys

def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load evaluation results from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"✗ Invalid JSON in: {filepath}")
        return None

def calculate_summary_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate summary metrics from results"""
    if not results:
        return {}
    
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))
    repeated = sum(1 for r in results if r.get('is_repeated', False))
    hallucinations = sum(1 for r in results if r.get('is_hallucination', False))
    
    # Calculate average metrics
    metrics = {
        'total_questions': total,
        'correct_answers': correct,
        'accuracy': correct / total if total > 0 else 0,
        'repeated_answers': repeated,
        'repetition_rate': repeated / total if total > 0 else 0,
        'hallucinations': hallucinations,
        'hallucination_rate': hallucinations / total if total > 0 else 0,
    }
    
    # Add average scores if available
    if results and 'metrics' in results[0]:
        avg_rouge1 = sum(r['metrics'].get('rouge1', 0) for r in results) / total
        avg_rouge2 = sum(r['metrics'].get('rouge2', 0) for r in results) / total
        avg_rougeL = sum(r['metrics'].get('rougeL', 0) for r in results) / total
        avg_semantic = sum(r['metrics'].get('semantic_similarity', 0) for r in results) / total
        avg_bleu = sum(r['metrics'].get('bleu_score', 0) for r in results) / total
        
        metrics.update({
            'avg_rouge1': avg_rouge1,
            'avg_rouge2': avg_rouge2,
            'avg_rougeL': avg_rougeL,
            'avg_semantic_similarity': avg_semantic,
            'avg_bleu_score': avg_bleu,
        })
    
    return metrics

def print_metrics_table(title: str, metrics: Dict[str, float]):
    """Print metrics in a formatted table"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    
    print(f"\n{'Metric':<35} {'Value':<15}")
    print('-'*60)
    
    # Core metrics
    print(f"{'Total Questions':<35} {metrics.get('total_questions', 0):<15}")
    print(f"{'Correct Answers':<35} {metrics.get('correct_answers', 0):<15}")
    print(f"{'Accuracy':<35} {metrics.get('accuracy', 0):<15.2%}")
    print(f"{'Repeated Answers':<35} {metrics.get('repeated_answers', 0):<15}")
    print(f"{'Repetition Rate':<35} {metrics.get('repetition_rate', 0):<15.2%}")
    print(f"{'Hallucinations':<35} {metrics.get('hallucinations', 0):<15}")
    print(f"{'Hallucination Rate':<35} {metrics.get('hallucination_rate', 0):<15.2%}")
    
    # Additional metrics
    if 'avg_rouge1' in metrics:
        print(f"\n{'--- Quality Metrics ---':<35}")
        print(f"{'Average ROUGE-1':<35} {metrics.get('avg_rouge1', 0):<15.4f}")
        print(f"{'Average ROUGE-2':<35} {metrics.get('avg_rouge2', 0):<15.4f}")
        print(f"{'Average ROUGE-L':<35} {metrics.get('avg_rougeL', 0):<15.4f}")
        print(f"{'Average Semantic Similarity':<35} {metrics.get('avg_semantic_similarity', 0):<15.4f}")
        print(f"{'Average BLEU Score':<35} {metrics.get('avg_bleu_score', 0):<15.4f}")

def print_comparison_table(base_metrics: Dict[str, float], ft_metrics: Dict[str, float]):
    """Print side-by-side comparison"""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: BASE MODEL vs FINE-TUNED MODEL")
    print('='*80)
    
    print(f"\n{'Metric':<30} {'Base':<15} {'Fine-tuned':<15} {'Change':<15}")
    print('-'*80)
    
    comparison_metrics = [
        ('accuracy', 'Accuracy', True),
        ('repetition_rate', 'Repetition Rate', False),
        ('hallucination_rate', 'Hallucination Rate', False),
        ('avg_rouge1', 'Avg ROUGE-1', True),
        ('avg_rouge2', 'Avg ROUGE-2', True),
        ('avg_rougeL', 'Avg ROUGE-L', True),
        ('avg_semantic_similarity', 'Avg Semantic Sim', True),
        ('avg_bleu_score', 'Avg BLEU', True),
    ]
    
    for metric_key, metric_name, higher_is_better in comparison_metrics:
        if metric_key in base_metrics and metric_key in ft_metrics:
            base_val = base_metrics[metric_key]
            ft_val = ft_metrics[metric_key]
            
            if higher_is_better:
                change = ft_val - base_val
                symbol = "↑" if change > 0 else "↓"
                change_pct = (change / base_val * 100) if base_val > 0 else 0
            else:
                change = base_val - ft_val
                symbol = "↓" if change > 0 else "↑"
                change_pct = (change / base_val * 100) if base_val > 0 else 0
            
            print(f"{metric_name:<30} {base_val:<15.4f} {ft_val:<15.4f} {symbol} {change_pct:>6.2f}%")

def show_sample_answers(results: List[Dict[str, Any]], num_samples: int = 5):
    """Show sample question-answer pairs"""
    print(f"\n{'='*80}")
    print(f"  SAMPLE ANSWERS (showing {min(num_samples, len(results))} of {len(results)})")
    print('='*80)
    
    for i, result in enumerate(results[:num_samples], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Question: {result.get('question', 'N/A')}")
        print(f"Expected: {result.get('expected_answer', 'N/A')}")
        print(f"Model:    {result.get('model_answer', 'N/A')}")
        print(f"Correct:  {'✓' if result.get('is_correct') else '✗'}")
        print(f"Repeated: {'✓' if result.get('is_repeated') else '✗'}")
        print(f"Hallucination: {'✓' if result.get('is_hallucination') else '✗'}")
        
        if 'metrics' in result:
            m = result['metrics']
            print(f"Metrics:  ROUGE-1={m.get('rouge1', 0):.3f}, "
                  f"BLEU={m.get('bleu_score', 0):.3f}, "
                  f"Semantic={m.get('semantic_similarity', 0):.3f}")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("  RAG EVALUATION RESULTS VIEWER")
    print("="*80)
    
    # File paths
    base_file = "evaluation/results/evaluation_results.json"
    ft_file = "evaluation/results/evaluation_results_finetuned.json"
    
    # Load results
    print("\nLoading results...")
    base_results = load_results(base_file)
    ft_results = load_results(ft_file)
    
    if not base_results and not ft_results:
        print("\n✗ No evaluation results found!")
        print("  Run evaluation first: python main.py --evaluate")
        sys.exit(1)
    
    # Display base model results
    if base_results:
        print(f"✓ Loaded base model results: {len(base_results)} questions")
        base_metrics = calculate_summary_metrics(base_results)
        print_metrics_table("BASE MODEL RESULTS", base_metrics)
    
    # Display fine-tuned model results
    if ft_results:
        print(f"\n✓ Loaded fine-tuned model results: {len(ft_results)} questions")
        ft_metrics = calculate_summary_metrics(ft_results)
        print_metrics_table("FINE-TUNED MODEL RESULTS", ft_metrics)
    
    # Display comparison if both exist
    if base_results and ft_results:
        print_comparison_table(base_metrics, ft_metrics)
    
    # Show sample answers
    if base_results:
        print("\n" + "="*80)
        response = input("\nShow sample answers from base model? (y/N): ")
        if response.lower() == 'y':
            num = input("How many samples? (default 5): ")
            num = int(num) if num.isdigit() else 5
            show_sample_answers(base_results, num)
    
    if ft_results:
        response = input("\nShow sample answers from fine-tuned model? (y/N): ")
        if response.lower() == 'y':
            num = input("How many samples? (default 5): ")
            num = int(num) if num.isdigit() else 5
            show_sample_answers(ft_results, num)
    
    print("\n" + "="*80)
    print("  Analysis Complete")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
