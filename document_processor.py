# document_processor.py (Enhanced)
from pathlib import Path
from typing import List
import re
import os

try:
    import pdfplumber
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False

try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    import nltk
    nltk.download("punkt")
    from nltk.tokenize import sent_tokenize


class DocumentProcessor:
    """Improved processor for extracting, cleaning, and chunking documents."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # 1️⃣ Extract text from PDF with fallback and cleaning
    # ------------------------------------------------------------------
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts clean text from PDF, removing noise."""
        text = ""

        if PDFPLUMBER_SUPPORT:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text(x_tolerance=1, y_tolerance=3)
                        if page_text:
                            page_text = self._clean_text(page_text)
                            text += f"[Page {i}]\n{page_text}\n\n"
            except Exception as e:
                print(f"[WARN] Failed to read {pdf_path}: {e}")
        else:
            print("⚠️ pdfplumber not installed. Install via: pip install pdfplumber")

        return text.strip()

    # ------------------------------------------------------------------
    # 2️⃣ Clean extracted text
    # ------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """Removes extra spaces, symbols, and repeated newlines."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
        text = text.replace("•", "-").replace("◦", "-")
        text = text.strip()
        return text

    # ------------------------------------------------------------------
    # 3️⃣ Sentence-aware chunking with overlap
    # ------------------------------------------------------------------
    def chunk_text(self, text: str) -> List[str]:
        """Splits text into overlapping, coherent chunks."""
        if not text or len(text) < self.chunk_size:
            return [text.strip()]

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) > self.chunk_size:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                # Overlap: retain last few sentences
                overlap_sentences = current_chunk[-3:]
                current_chunk = overlap_sentences.copy()
                current_length = len(" ".join(current_chunk).split())

            current_chunk.append(sentence)
            current_length += len(words)

        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ------------------------------------------------------------------
    # 4️⃣ Load and process directory (supports PDFs)
    # ------------------------------------------------------------------
    def load_documents_from_directory(self, directory: str) -> List[str]:
        """Loads all PDFs and returns a list of processed text chunks."""
        directory_path = Path(directory)
        all_chunks = []

        if not directory_path.exists():
            print(f"⚠️ Directory not found: {directory}")
            return []

        pdf_files = list(directory_path.glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️ No PDF files found in {directory}")
            return []

        for pdf_path in pdf_files:
            print(f"\n📘 Processing: {pdf_path.name}")
            text = self.extract_text_from_pdf(str(pdf_path))

            if not text:
                print(f"  [WARN] No text extracted from {pdf_path.name}")
                continue

            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)
            print(f"  [OK] {len(chunks)} chunks created from {pdf_path.name}")

        print(f"\n✅ Total {len(all_chunks)} chunks generated from all PDFs.")
        return all_chunks
