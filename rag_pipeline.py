from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List
from pathlib import Path
from datetime import datetime
import csv


class RAGPipeline:
    def __init__(self, config=None, document_chunks=None, model_path=None):
        """
        Retrieval-Augmented Generation (RAG) pipeline using
        fine-tuned FLAN-T5 (if available) and MiniLM as embedder.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.document_chunks = document_chunks or []

        # ------------------------------------------------------------------
        # 🔹 Load Fine-tuned or Base Model Automatically
        # ------------------------------------------------------------------
        default_model_name = "google/flan-t5-small"
        finetuned_model_path = model_path or Path("models/finetuned")

        if finetuned_model_path.exists():
            self.model_name = str(finetuned_model_path)
            print(f"🚀 Loading fine-tuned model from: {finetuned_model_path}")
            self.is_finetuned = True
        else:
            self.model_name = default_model_name
            print(f"⚠️ Fine-tuned model not found at {finetuned_model_path}, using base model instead.")
            self.is_finetuned = False

        # ------------------------------------------------------------------
        # 🔹 Load Generator Model
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        # Create generator pipeline
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

        # ------------------------------------------------------------------
        # 🔹 Load Embedding Model
        # ------------------------------------------------------------------
        print("🔎 Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        print(f"✅ RAG pipeline initialized using: {self.model_name} ({'Fine-tuned' if self.is_finetuned else 'Base'})\n")

        # ------------------------------------------------------------------
        # 🔹 Log model usage for experiment tracking
        # ------------------------------------------------------------------
        self._log_model_usage()

    # ============================================================
    #  📘 EMBEDDING + RETRIEVAL
    # ============================================================
    def embed_text(self, texts: List[str]) -> torch.Tensor:
        """Generate dense vector embeddings for given texts."""
        return self.embedding_model.encode(texts, convert_to_tensor=True)

    def retrieve_context(self, query: str, documents: List[str] = None, top_k: int = 3) -> List[str]:
        """Retrieve the top-k most relevant chunks for a given query."""
        documents = documents or self.document_chunks
        if not documents:
            print("⚠️ No documents loaded into RAG.")
            return []

        query_embedding = self.embed_text([query])
        doc_embeddings = self.embed_text(documents)

        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

        similarities = torch.mm(query_embedding, doc_embeddings.transpose(0, 1))[0]
        top_k = min(top_k, len(documents))
        _, top_indices = torch.topk(similarities, k=top_k)

        return [documents[idx] for idx in top_indices]

    # ============================================================
    #  💬 GENERATION
    # ============================================================
    def generate_response(self, question: str, contexts: List[str] = None) -> str:
        """Generate an answer strictly based on the given context."""
        try:
            if not contexts:
                contexts = self.retrieve_context(question, top_k=3)

            context_text = "\n\n".join(contexts) if contexts else "No context found."

            # Truncate context to fit model window
            MAX_CONTEXT_TOKENS = 450
            tokenized_ctx = self.tokenizer(
                context_text, truncation=True, max_length=MAX_CONTEXT_TOKENS
            )
            context_text = self.tokenizer.decode(tokenized_ctx["input_ids"], skip_special_tokens=True)

            # Construct input prompt
            prompt = (
                "You are a factual question answering assistant.\n"
                "Use only the context provided to answer accurately.\n"
                "If the answer isn't present, reply 'Not found in document.'\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\nAnswer:"
            )

            input_len = len(self.tokenizer(prompt)["input_ids"])
            print(f"[DEBUG] Input tokens: {input_len}")
            print("[DEBUG] Context snippet:", context_text[:250].replace("\n", " ") + "...\n")

            outputs = self.generator(
                prompt,
                max_new_tokens=120,
                num_return_sequences=1,
                clean_up_tokenization_spaces=True
            )
            answer = outputs[0]["generated_text"].strip()

            # Validate grounding
            if not self._validate_grounding(answer, context_text):
                if answer.lower() not in context_text.lower()[:400]:
                    answer = "Not found in document."

            return answer

        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return "Error while generating response."

    # ============================================================
    #  ✅ VALIDATION
    # ============================================================
    def _validate_grounding(self, answer: str, context: str, threshold: float = 0.25) -> bool:
        """Check if answer text aligns semantically with context."""
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, answer.lower(), context.lower()).ratio()
        return ratio >= threshold

    # ============================================================
    #  📊 MODEL USAGE LOGGING
    # ============================================================
    def _log_model_usage(self):
        """Log model usage info to metrics_log.csv"""
        metrics_file = Path("evaluation") / "metrics_log.csv"
        metrics_file.parent.mkdir(exist_ok=True)

        fieldnames = ["timestamp", "model_name", "mode", "device"]

        file_exists = metrics_file.exists()
        with open(metrics_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": self.model_name,
                "mode": "Fine-tuned" if self.is_finetuned else "Base",
                "device": self.device
            })

        print(f"🧾 Logged model usage to: {metrics_file}")

    # ============================================================
    #  🔁 PIPELINE RUNNER
    # ============================================================
    def run_pipeline(self, questions: List[str]) -> List[dict]:
        """Run RAG pipeline for a list of questions."""
        results = []
        for q in questions:
            ans = self.generate_response(q)
            results.append({"question": q, "answer": ans})
        return results
