# ⚖️ Legal QA - Question Answering for Legal Documents

A transformer-based question answering system for legal documents. Supports fine-tuning BERT, DistilBERT, and RoBERTa models.

## Project Structure

```
legal-qa/
├── scripts/
│   ├── bert.py              # BERT training script
│   ├── distilbert.py        # DistilBERT training script
│   ├── roberta.py           # RoBERTa training script
│   └── evaluate.py          # Model evaluation script
├── data/
│   ├── train.json           # Training dataset
│   └── test.json            # Test/evaluation dataset
├── models/
│   ├── bert/                # Trained BERT model
│   ├── distilbert/          # Trained DistilBERT model
│   └── roberta/             # Trained RoBERTa model
├── results/
│   ├── bert/                # BERT evaluation results
│   ├── distilbert/          # DistilBERT evaluation results
│   └── roberta/             # RoBERTa evaluation results
├── checkpoints/             # Training checkpoints
├── nltk_data/               # NLTK data (auto-downloaded)
├── requirements.txt         # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on A6000)
- ~10GB VRAM minimum (48GB recommended for large batch sizes)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mwnthainarzary/legal-qa.git
cd legal-qa
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training scripts from the project root:
```bash
# Train BERT
python scripts/bert.py

# Train DistilBERT
python scripts/distilbert.py

# Train RoBERTa
python scripts/roberta.py
```

The scripts will:
1. Load and preprocess the QA dataset
2. Fine-tune the model for question answering
3. Save the best model to `models/<model_name>/`

### Evaluation

Run the evaluation script:
```bash
# Evaluate BERT
python scripts/evaluate.py --model_path ./models/bert/ --test_file ./data/test.json --output_file ./results/bert/evaluation.json

# Evaluate DistilBERT
python scripts/evaluate.py --model_path ./models/distilbert/ --test_file ./data/test.json --output_file ./results/distilbert/evaluation.json

# Evaluate RoBERTa
python scripts/evaluate.py --model_path ./models/roberta/ --test_file ./data/test.json --output_file ./results/roberta/evaluation.json
```

#### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to the trained model directory |
| `--test_file` | (required) | Path to the test dataset (JSON) |
| `--batch_size` | 8 | Batch size for evaluation |
| `--max_length` | 512 | Maximum sequence length |
| `--output_file` | `evaluation_results.json` | Output file for metrics |

#### Evaluation Metrics

The evaluation script computes:

**Position-based Metrics:**
- Start Position Accuracy & F1
- End Position Accuracy & F1

**Text-based Metrics:**
- **Exact Match**: Percentage of predictions matching ground truth exactly
- **BLEU Score**: N-gram overlap between prediction and reference
- **BERT Score**: Semantic similarity using BERT embeddings (Precision/Recall/F1)
- **Cosine Similarity**: Sentence embedding similarity using Sentence Transformers

#### Evaluation Outputs

Results are saved to the `results/<model>/` directory:
1. `evaluation.json` - All evaluation metrics
2. `evaluation_predictions.json` - Model predictions with:
   - Question
   - Context
   - Ground truth answer
   - Predicted answer
   - Exact match flag

### Configuration

Key training parameters in training scripts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 25 | Number of training epochs |
| `per_device_train_batch_size` | 128 | Batch size per GPU |
| `max_length` | 384 | Maximum sequence length |
| `bf16` | True | Use BF16 mixed precision |

### GPU Optimizations

The script is optimized for NVIDIA Ampere GPUs (A6000, A100, RTX 30/40 series):
- TF32 enabled for faster matrix operations
- BF16 mixed precision training
- Fused AdamW optimizer
- Parallel data loading with prefetching

## Data Format

The dataset should be in JSON format:

```json
[
  [
    {
      "context": "Legal document text...",
      "questions": [
        {
          "question": "What is the penalty for...?",
          "answer": {
            "text": "The penalty is...",
            "answer_start": 42
          }
        }
      ]
    }
  ]
]
```

## Models

| Model | Base | Parameters | Description |
|-------|------|------------|-------------|
| BERT | `bert-base-cased` | 110M | Standard BERT model |
| DistilBERT | `distilbert-base-cased` | 66M | 40% smaller, 60% faster |
| RoBERTa | `roberta-base` | 125M | Robustly optimized BERT |

**Task**: Extractive Question Answering  
**Output**: Start and end positions of the answer span in the context

## Results

| Metric | BERT | DistilBERT | RoBERTa |
|--------|------|------------|---------|
| Start Position Accuracy | 0.5308 | 0.5232 | 0.6128 |
| End Position Accuracy | 0.6051 | 0.5876 | 0.6895 |
| Exact Match | 0.1838 | 0.1791 | 0.4837 |
| BLEU Score | 0.4117 | 0.3908 | 0.5699 |
| BERT Score F1 | 0.9327 | 0.9305 | 0.9562 |
| Cosine Similarity | 0.7967 | 0.7845 | 0.8545 |

## License

See [LICENSE](LICENSE) for details.
