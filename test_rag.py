import os
import sys
from pathlib import Path
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline
from config import Config

def test_rag_with_pdf(pdf_path: str, question: str):
    """Test the RAG pipeline with a specific PDF and question"""
    print("=" * 80)
    print(f"TESTING RAG PIPELINE WITH: {os.path.basename(pdf_path)}")
    print("=" * 80)
    
    # Initialize document processor
    doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Load and process the PDF
    print(f"\nProcessing PDF: {pdf_path}")
    text = doc_processor.extract_text_from_pdf(pdf_path)
    
    if not text.strip():
        print("Error: Could not extract text from PDF")
        return
    
    # Chunk the text
    chunks = doc_processor.chunk_text(text)
    print(f"Extracted {len(chunks)} chunks from the document")
    
    if not chunks:
        print("Error: No chunks created from the document")
        return
    
    # Initialize RAG pipeline with the chunks
    print("\nInitializing RAG pipeline...")
    config = Config()
    rag_pipeline = RAGPipeline(config=config, document_chunks=chunks)
    
    # Test retrieval
    print(f"\nTesting retrieval for question: {question}")
    contexts = rag_pipeline.retrieve_context(question, top_k=2)
    
    if not contexts:
        print("No relevant context found for the question.")
        return
    
    print("\nTop context chunks:")
    for i, ctx in enumerate(contexts, 1):
        print(f"\n--- Context {i} (Length: {len(ctx)} chars) ---")
        print(ctx[:500] + "..." if len(ctx) > 500 else ctx)
    
    # Generate response
    print("\nGenerating response...")
    response = rag_pipeline.generate_response(question)
    
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("-" * 80)
    print("RESPONSE:", response)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    # Default PDF path and question
    pdf_path = os.path.join("data", "raw", "Nvidia Corporation.pdf")
    question = "where is headquarters of Nvidia?"
    
    # Allow overriding with command line arguments
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        question = " ".join(sys.argv[2:])
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please provide the path to the Nvidia Corporation PDF as an argument.")
        sys.exit(1)
    
    test_rag_with_pdf(pdf_path, question)
