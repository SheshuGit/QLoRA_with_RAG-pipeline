from typing import List, Dict
import json
from pathlib import Path
from dataclasses import dataclass, field
from config import Config
from rouge_score import rouge_scorer
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------
# NLTK setup
# ----------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class EvaluationMetrics:
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    semantic_similarity: float = 0.0
    bleu_score: float = 0.0

@dataclass
class EvaluationResult:
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    is_repeated: bool
    is_hallucination: bool
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)

class RAGEvaluator:
    def __init__(self):
        self.config = Config()
        self.results_dir = Path(self.config.EVAL_DIR) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)

    # ----------------------------------------------------------
    # Core metric calculators
    # ----------------------------------------------------------
    def _calculate_semantic_similarity(self, ref: str, pred: str) -> float:
        """Compute cosine similarity between reference and predicted embeddings."""
        ref_emb = self.embedding_model.encode([ref])[0]
        pred_emb = self.embedding_model.encode([pred])[0]
        score = np.dot(ref_emb, pred_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(pred_emb))
        return float(score)

    def _calculate_bleu(self, ref: str, pred: str) -> float:
        """Calculate BLEU score with smoothing for short answers."""
        smoothie = SmoothingFunction().method1
        ref_tokens = [word_tokenize(ref.lower())]
        pred_tokens = word_tokenize(pred.lower())
        try:
            return float(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie))
        except ZeroDivisionError:
            return 0.0

    # ----------------------------------------------------------
    # Evaluation logic
    # ----------------------------------------------------------
    def evaluate_response(
        self,
        question: str,
        expected_answer: str,
        model_answer: str,
        previous_answers: List[str] = None
    ) -> EvaluationResult:
        """Evaluate a single Q&A pair."""
        previous_answers = previous_answers or []

        # --- correctness ---
        is_correct = self._check_answer_correctness(expected_answer, model_answer)
        is_repeated = model_answer in previous_answers
        is_hallucination = self._check_hallucination(question, model_answer)

        # --- scores ---
        rouge_scores = self.rouge.score(expected_answer, model_answer)
        semantic_sim = self._calculate_semantic_similarity(expected_answer, model_answer)
        bleu_score = self._calculate_bleu(expected_answer, model_answer)

        metrics = EvaluationMetrics(
            rouge1=float(rouge_scores['rouge1'].fmeasure),
            rouge2=float(rouge_scores['rouge2'].fmeasure),
            rougeL=float(rouge_scores['rougeL'].fmeasure),
            semantic_similarity=float(semantic_sim),
            bleu_score=float(bleu_score)
        )

        return EvaluationResult(
            question=str(question),
            expected_answer=str(expected_answer),
            model_answer=str(model_answer),
            is_correct=bool(is_correct),
            is_repeated=bool(is_repeated),
            is_hallucination=bool(is_hallucination),
            metrics=metrics
        )

    def _check_answer_correctness(self, expected: str, actual: str) -> bool:
        """Check correctness via semantic similarity."""
        expected, actual = expected.lower().strip(), actual.lower().strip()
        if not expected or not actual:
            return False
        sim = self._calculate_semantic_similarity(expected, actual)
        return bool(sim > 0.7)

    def _check_hallucination(self, question: str, answer: str) -> bool:
        """Detect hallucinations via generic non-answers or low semantic relation."""
        hallucination_phrases = [
            "i don't know", "not found", "not in document", "no information",
            "i can't answer", "not mentioned", "not provided", "unclear"
        ]

        if any(p in answer.lower() for p in hallucination_phrases):
            return True

        q_emb = self.embedding_model.encode([question])[0]
        a_emb = self.embedding_model.encode([answer])[0]
        sim = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
        return bool(sim < 0.25)

    # ----------------------------------------------------------
    # Save & aggregate results
    # ----------------------------------------------------------
    def save_results(self, results: List[EvaluationResult], filename="evaluation_results.json") -> str:
        """Save evaluation results to JSON file safely."""
        result_data = []

        for r in results:
            result_data.append({
                "question": str(r.question),
                "expected_answer": str(r.expected_answer),
                "model_answer": str(r.model_answer),
                "is_correct": bool(r.is_correct),
                "is_repeated": bool(r.is_repeated),
                "is_hallucination": bool(r.is_hallucination),
                "metrics": {
                    "rouge1": float(r.metrics.rouge1),
                    "rouge2": float(r.metrics.rouge2),
                    "rougeL": float(r.metrics.rougeL),
                    "semantic_similarity": float(r.metrics.semantic_similarity),
                    "bleu_score": float(r.metrics.bleu_score)
                }
            })

        output_path = self.results_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved evaluation results to {output_path}")
        return str(output_path)

    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Aggregate and average all evaluation metrics."""
        total = len(results)
        if total == 0:
            return {}

        correct = sum(1 for r in results if r.is_correct)
        repeated = sum(1 for r in results if r.is_repeated)
        hallucinated = sum(1 for r in results if r.is_hallucination)

        avg_rouge1 = float(np.mean([r.metrics.rouge1 for r in results]))
        avg_rouge2 = float(np.mean([r.metrics.rouge2 for r in results]))
        avg_rougeL = float(np.mean([r.metrics.rougeL for r in results]))
        avg_semantic = float(np.mean([r.metrics.semantic_similarity for r in results]))
        avg_bleu = float(np.mean([r.metrics.bleu_score for r in results]))

        return {
            "accuracy": float(correct / total),
            "repetition_rate": float(repeated / total),
            "hallucination_rate": float(hallucinated / total),
            "avg_rouge1": avg_rouge1,
            "avg_rouge2": avg_rouge2,
            "avg_rougeL": avg_rougeL,
            "avg_semantic_similarity": avg_semantic,
            "avg_bleu_score": avg_bleu,
            "total_questions": int(total)
        }
