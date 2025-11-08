# main.py
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from data_loader import S3DataLoader
from rag_pipeline import RAGPipeline
from evaluation import RAGEvaluator, EvaluationResult
from evaluation import RAGEvaluator, EvaluationResult, EvaluationMetrics

from config import Config
from document_processor import DocumentProcessor

def load_golden_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate the golden dataset"""
    try:
        with open(file_path, 'r') as f:
            dataset = json.load(f)
        
        # Validate dataset format
        required_keys = {'question', 'answer'}
        for i, item in enumerate(dataset):
            if not all(key in item for key in required_keys):
                raise ValueError(f"Missing required keys in item {i}: {item.keys()}")
        
        return dataset
    
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Golden dataset not found at {file_path}. "
            "Please create a JSON file with 'question' and 'answer' fields."
        )
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")

def print_comparison(base_metrics: Dict[str, float], finetuned_metrics: Dict[str, float]):
    """Print side-by-side comparison of base vs fine-tuned metrics"""
    print(f"\n{'Metric':<30} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for metric in base_metrics.keys():
        if metric == "total_questions":
            continue
            
        base_val = base_metrics[metric]
        ft_val = finetuned_metrics[metric]
        
        # Calculate improvement
        if isinstance(base_val, (int, float)):
            if metric in ["repetition_rate", "hallucination_rate"]:
                # Lower is better for these metrics
                improvement = base_val - ft_val
                improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
                symbol = "↓" if improvement > 0 else "↑"
            else:
                # Higher is better for accuracy and other metrics
                improvement = ft_val - base_val
                improvement_pct = (improvement / base_val * 100) if base_val > 0 else 0
                symbol = "↑" if improvement > 0 else "↓"
            
            print(f"{metric:<30} {base_val:<15.4f} {ft_val:<15.4f} {symbol} {improvement_pct:>6.2f}%")
        else:
            print(f"{metric:<30} {base_val:<15} {ft_val:<15}")
    
    print("-" * 80)
    print(f"\n{'Total Questions Evaluated:':<30} {base_metrics.get('total_questions', 0)}")
def run_evaluation(
    rag_pipeline: RAGPipeline,
    evaluator: RAGEvaluator,
    dataset: List[Dict[str, Any]],
    output_file: str = "evaluation_results.json"
) -> Dict[str, float]:
    """Run evaluation on the RAG pipeline with top 1 context only and export to JSON + CSV"""
    results: List[Dict[str, Any]] = []
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)

    print(f"\n🧠 Starting evaluation on {len(dataset)} questions...\n")

    for i, item in enumerate(dataset, start=1):
        question = item.get("question", "").strip()
        expected_answer = item.get("answer", "").strip()

        if not question or not expected_answer:
            print(f"[WARN] Skipping item {i}: Missing question or answer")
            continue

        try:
            # Retrieve relevant context
            contexts = (
                rag_pipeline.retrieve_context(question, rag_pipeline.document_chunks, top_k=3)
                if rag_pipeline.document_chunks
                else [item.get("context", "")]
            )

            # Generate model response
            model_answer = rag_pipeline.generate_response(question, contexts)
            if not model_answer:
                model_answer = "No response generated."

            # Evaluate the response
            result = evaluator.evaluate_response(
                question=question,
                expected_answer=expected_answer,
                model_answer=model_answer
            )

            # Format results
            formatted_result = {
                "question": question,
                "expected": expected_answer,
                "model": model_answer,
                "correct": bool(result.is_correct),
                "repeated": bool(result.is_repeated),
                "hallucination": bool(result.is_hallucination),
                "rouge1": float(result.metrics.rouge1),
                "bleu": float(result.metrics.bleu_score),
                "semantic": float(result.metrics.semantic_similarity)
            }

            results.append(formatted_result)
            print(f"[{i}/{len(dataset)}] ✅ Processed: {question[:70]}")

        except Exception as e:
            print(f"[{i}/{len(dataset)}] ⚠️ Error processing: {question[:50]} → {e}")
            continue

    print(f"\n🧾 Total valid results: {len(results)} / {len(dataset)}")

    # --- Save results to JSON ---
    json_path = eval_dir / output_file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to JSON: {json_path}")

    # --- Save results to CSV ---
    import csv
    csv_path = eval_dir / output_file.replace(".json", ".csv")
    fieldnames = ["question", "expected", "model", "correct", "repeated", "hallucination", "rouge1", "bleu", "semantic"]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"📊 Results also saved to CSV: {csv_path}")

    # --- If no results, warn and exit ---
    if not results:
        print("⚠️ No valid results generated. Please check dataset or context retrieval.")
        return {}

    # --- Compute aggregate metrics ---
    metrics = evaluator.calculate_metrics([
        EvaluationResult(
            question=r["question"],
            expected_answer=r["expected"],
            model_answer=r["model"],
            is_correct=r["correct"],
            is_repeated=r["repeated"],
            is_hallucination=r["hallucination"],
            metrics=EvaluationMetrics(
                rouge1=r["rouge1"],
                rouge2=0,
                rougeL=0,
                semantic_similarity=r["semantic"],
                bleu_score=r["bleu"]
            )
        )
        for r in results
    ])

    # --- Log metrics summary to CSV ---
    import pandas as pd
    from datetime import datetime

    metrics_file = eval_dir / "metrics_log.csv"
    metrics_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": getattr(rag_pipeline, "model_name", "unknown-model"),
        **metrics
    }

    df = pd.DataFrame([metrics_row])
    if metrics_file.exists():
        df_existing = pd.read_csv(metrics_file)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(metrics_file, index=False)
    print(f"📈 Metrics summary logged to: {metrics_file}")

    return metrics



def main():
    def safe_print(*args, **kwargs):
        """Safely print text, handling Windows console encoding issues"""
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            # Fallback to ASCII-only output
            cleaned_args = [str(arg).encode('ascii', errors='replace').decode('ascii') for arg in args]
            print(*cleaned_args, **kwargs)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Pipeline with Evaluation")
    parser.add_argument("--download-data", action="store_true", help="Download data from S3")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the model")
    parser.add_argument("--dataset", type=str, default="golden_data.json", 
                       help="Path to golden dataset JSON file")
    args = parser.parse_args()

    config = Config()
    
    # Create necessary directories
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.EVAL_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    data_loader = S3DataLoader()
    doc_processor = DocumentProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    # Initialize document_chunks as empty list
    document_chunks = []
    
    # Download data if requested
    if args.download_data:
        # Function to safely print with fallback for Windows console
        def safe_print(*args, **kwargs):
            try:
                print(*args, **kwargs)
            except UnicodeEncodeError:
                # Fallback to ASCII-only output
                cleaned_args = [str(arg).encode('ascii', errors='replace').decode('ascii') for arg in args]
                print(*cleaned_args, **kwargs)
        
        # Initialize document processor and load documents
        doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        safe_print("\n" + "="*80)
        safe_print("LOADING DOCUMENTS FOR RAG CONTEXT")
        safe_print("="*80)
        safe_print()
        
        # Load and process documents
        raw_data_dir = os.path.join(config.DATA_DIR, 'raw')
        document_chunks = []
        
        # Create raw directory if it doesn't exist
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # Check if we have any PDF files in the raw directory
        pdf_files = list(Path(raw_data_dir).glob('*.pdf'))
        
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF files in {raw_data_dir}")
            document_chunks = doc_processor.load_documents_from_directory(raw_data_dir)
            print(f"Processed {len(document_chunks)} document chunks from PDFs")
        else:
            print(f"No PDF files found in {raw_data_dir}. Please add PDF documents to this directory for better context.")
            print("The system will rely on the golden dataset context only.")  # Run with --download-data to fetch from S3, or add documents manually")
            print("  RAG will use golden dataset context as fallback.")
    
    # Load golden dataset
    try:
        safe_print(f"\nLoading golden dataset from {args.dataset}...")
        golden_dataset = load_golden_dataset(args.dataset)
        safe_print(f"[OK] Loaded {len(golden_dataset)} examples from golden dataset")
    except Exception as e:
        safe_print(f"[ERROR] Error loading golden dataset: {str(e)}")
        return
    
    # Initialize RAG pipeline with document chunks
    print("\nInitializing RAG pipeline...")
    config = Config()
    try:
        # Initialize the RAG pipeline with our documents
        rag_pipeline = RAGPipeline(config=config, document_chunks=document_chunks)
        
        # Test the document retrieval
        if document_chunks:
            print(f"[OK] RAG pipeline initialized with {len(document_chunks)} document chunks")
            
            # Test retrieval with a sample query
            test_query = "What is Nvidia known for?"
            test_contexts = rag_pipeline.retrieve_context(test_query, top_k=1)
            if test_contexts:
                print(f"  Test retrieval successful. Sample context length: {len(test_contexts[0])} chars")
            else:
                print("  [WARN] No context could be retrieved. The system may not have relevant information.")
        else:
            print("[OK] RAG pipeline initialized (will use golden dataset context only)")
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG pipeline: {str(e)}")
        print("Falling back to golden dataset context only...")
        rag_pipeline = RAGPipeline(config=config, document_chunks=[])
    
    evaluator = RAGEvaluator()
    
    # Run evaluation if requested
    if args.evaluate:
        print("Running evaluation...")
        metrics = run_evaluation(
            rag_pipeline=rag_pipeline,
            evaluator=evaluator,
            dataset=golden_dataset,
            output_file="evaluation_results.json"
        )
    
    # Fine-tune if requested
    if args.fine_tune:
        print("Starting fine-tuning...")
        from fine_tuning import fine_tune_model
        fine_tune_model()
        
        # If both evaluate and fine-tune are specified, run evaluation after fine-tuning
        if args.evaluate:
            print("\n" + "="*80)
            print("EVALUATING FINE-TUNED MODEL")
            print("="*80)
            
            # Reload the fine-tuned model with same document chunks
            rag_pipeline_finetuned = RAGPipeline(
                model_path=os.path.join(config.MODELS_DIR, "finetuned"),
                document_chunks=document_chunks
            )
            metrics_after = run_evaluation(
                rag_pipeline=rag_pipeline_finetuned,
                evaluator=evaluator,
                dataset=golden_dataset,
                output_file="evaluation_results_finetuned.json"
            )
            
            # Compare base vs fine-tuned metrics
            print("\n" + "="*80)
            print("COMPARISON: BASE MODEL vs FINE-TUNED MODEL")
            print("="*80)
            print_comparison(metrics, metrics_after)

if __name__ == "__main__":
    main()