# Legal QA - BERT Question Answering

A BERT-based question answering system for legal documents. Fine-tunes `bert-base-cased` on a custom legal QA dataset.

## Project Structure

```
legal-qa/
├── bert.py                  # Main training script
├── evaluate.py              # Model evaluation script
├── data/
│   ├── train.json           # Training dataset
│   └── test.json            # Test/evaluation dataset
├── bert_qa_best/            # Best trained model
├── bert_qa_results/         # Training checkpoints
├── nltk_data/               # NLTK data (auto-downloaded)
├── evaluation_results.json  # Evaluation metrics
├── evaluation_results_predictions.json  # Model predictions
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

Run the training script:
```bash
python bert.py
python distilbert.py
python roberta.py
```

The script will:
1. Load and preprocess the QA dataset
2. Fine-tune BERT for question answering
3. Save the best model to `bert_qa_best/`

### Evaluation

Run the evaluation script:
```bash
python evaluate.py --model_path ./bert_qa_best/ --test_file ./data/test.json
python evaluate.py --model_path ./distilbert_qa_best/ --test_file ./data/test.json --output_file distilbert_evaluation_results.json
python evaluate.py --model_path ./roberta_qa_best/ --test_file ./data/test.json --output_file roberta_evaluation_results.json
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

1. `evaluation_results.json` - All evaluation metrics
2. `evaluation_results_predictions.json` - Model predictions with:
   - Question
   - Context
   - Ground truth answer
   - Predicted answer
   - Exact match flag

### Configuration

Key training parameters in `bert.py`:

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

## Model

- **Base Model**: `bert-base-cased`
- **Task**: Extractive Question Answering
- **Output**: Start and end positions of the answer span in the context

## Results

| Metric | Score |
|--------|-------|
| Start Position Accuracy | 0.5308 |
| End Position Accuracy | 0.6051 |
| Exact Match | 0.1838 |
| BLEU Score | 0.4117 |
| BERT Score F1 | 0.9327 |
| Cosine Similarity | 0.7967 |

## License

See [LICENSE](LICENSE) for details.
