# RAG Pipeline with Fine-Tuning & Evaluation

A comprehensive Retrieval-Augmented Generation (RAG) pipeline with QLoRA fine-tuning capabilities and robust evaluation metrics.

## 🚀 Features

- **RAG Pipeline**: Semantic search with sentence transformers for context retrieval
- **QLoRA Fine-Tuning**: Efficient 4-bit quantized fine-tuning with LoRA adapters
- **Comprehensive Evaluation**: ROUGE, BLEU, semantic similarity, hallucination detection
- **S3 Integration**: Load documents directly from AWS S3
- **Golden Dataset**: Pre-built evaluation dataset with 100+ Nvidia-related Q&A pairs

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for fine-tuning)
- AWS account with S3 access (optional, for data loading)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd llm_task_windsurf
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your AWS credentials
# AWS_ACCESS_KEY=your_aws_access_key_here
# AWS_SECRET_KEY=your_aws_secret_key_here
# S3_BUCKET=your_s3_bucket_name_here
```

## 📁 Project Structure

```
llm_task_windsurf/
├── main.py                 # Main entry point
├── config.py              # Configuration settings
├── data_loader.py         # S3 data loading utilities
├── rag_pipeline.py        # RAG implementation
├── fine_tuning.py         # QLoRA fine-tuning logic
├── evaluation.py          # Evaluation metrics and scoring
├── golden_data.json       # Golden dataset for evaluation
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── data/                 # Data directory
│   ├── raw/             # Raw documents
│   └── processed/       # Processed documents
├── models/              # Model checkpoints
│   ├── base/           # Base models
│   └── finetuned/      # Fine-tuned models
└── evaluation/         # Evaluation results
    └── results/        # Detailed evaluation outputs
```

## 🎯 Usage

### Quick Start: Complete Workflow

Run the entire pipeline automatically (recommended for first-time users):

```bash
python run_complete_workflow.py
```

This will:
1. ✅ Check prerequisites
2. ✅ Prepare training data from golden dataset
3. ✅ Evaluate base model
4. ✅ Fine-tune the model (with confirmation)
5. ✅ Evaluate fine-tuned model
6. ✅ Display side-by-side comparison
7. ✅ Save all results

### Manual Usage

### 1. Evaluate Base Model

Run evaluation on the base model using the golden dataset:

```bash
python main.py --evaluate
```

This will:
- Load the `golden_data.json` dataset
- Initialize the RAG pipeline with TinyLlama
- Generate answers for each question
- Calculate comprehensive metrics (accuracy, ROUGE, BLEU, semantic similarity)
- Save results to `evaluation/results/evaluation_results.json`

### 2. Download Data from S3

**Important**: The RAG pipeline retrieves context from documents stored in `data/raw/`. You have two options:

**Option A: Download from S3** (if you have S3 configured)
```bash
python main.py --download-data --evaluate
```

**Option B: Use local documents** (for testing without S3)
```bash
# Copy your documents to data/raw/
mkdir -p data/raw
cp sample_document.txt data/raw/

# Then run evaluation
python main.py --evaluate
```

The RAG pipeline will:
1. Load all documents from `data/raw/`
2. Chunk them into smaller pieces (1000 chars with 200 overlap)
3. Use semantic search to find relevant chunks for each question
4. Generate answers based on retrieved context

**Fallback**: If no documents are found, the RAG will use the context field from the golden dataset.

### 3. Fine-Tune the Model

Fine-tune the model using QLoRA (requires `data/rag_train.json`):

```bash
python main.py --fine-tune
```

**Note**: You need to create a training dataset at `data/rag_train.json` with the same format as `golden_data.json`.

### 4. Evaluate After Fine-Tuning

Run both fine-tuning and evaluation:

```bash
python main.py --fine-tune --evaluate
```

This will:
1. Fine-tune the model
2. Evaluate the base model
3. Evaluate the fine-tuned model
4. Save separate results for comparison

### 5. Custom Dataset

Use a custom evaluation dataset:

```bash
python main.py --evaluate --dataset path/to/your/dataset.json
```

### 6. View and Analyze Results

After running evaluations, view detailed results and comparisons:

```bash
python view_results.py
```

This interactive tool will:
- Display metrics for base and fine-tuned models
- Show side-by-side comparison with improvement percentages
- Allow browsing sample question-answer pairs
- Highlight correct answers, repetitions, and hallucinations

## 📊 Evaluation Metrics

The evaluation system tracks:

- **Accuracy**: Semantic similarity-based correctness (threshold: 0.7)
- **Repetition Rate**: Percentage of repeated answers
- **Hallucination Rate**: Detection of non-answers or irrelevant responses
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L for n-gram overlap
- **BLEU Score**: Machine translation quality metric
- **Semantic Similarity**: Cosine similarity of sentence embeddings

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
```

## 📝 Golden Dataset Format

The `golden_data.json` file should follow this structure:

```json
[
  {
    "question": "Your question here?",
    "context": "Relevant context with citations [cite: 1]",
    "answer": "Expected answer [cite: 1]"
  }
]
```

## 🧪 QLoRA Fine-Tuning Details

The fine-tuning uses:
- **4-bit quantization** (NF4) for memory efficiency
- **LoRA adapters** with rank=16, alpha=32
- **Target modules**: q_proj, v_proj (attention layers)
- **Batch size**: 4 with gradient accumulation (effective batch size: 16)
- **Learning rate**: 2e-4 with warmup

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` in `fine_tuning.py`
- Reduce `CHUNK_SIZE` in `config.py`
- Use a smaller model

### Missing NLTK Data
```python
import nltk
nltk.download('punkt')
```

### S3 Access Issues
- Verify AWS credentials in `.env`
- Check S3 bucket permissions
- Ensure boto3 is installed

## 📈 Performance Tips

1. **Use GPU**: Significantly faster for both inference and fine-tuning
2. **Batch Processing**: Process multiple questions in parallel
3. **Cache Embeddings**: Store document embeddings to avoid recomputation
4. **Optimize TOP_K**: Balance between context quality and speed

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- TinyLlama for the base language model
- Sentence Transformers for embedding models
- Hugging Face for the transformers library
- PEFT library for QLoRA implementation
