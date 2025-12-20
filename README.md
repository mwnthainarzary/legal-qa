# Legal QA - BERT Question Answering

A BERT-based question answering system for legal documents. Fine-tunes `bert-base-cased` on a custom legal QA dataset.

## Project Structure

```
legal-qa/
├── bert.py              # Main training script
├── data/
│   ├── train.json       # Training dataset
│   └── test.json        # Test/evaluation dataset
├── bert_qa_results/     # Model outputs (generated)
├── requirements.txt     # Python dependencies
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
git clone <repo-url>
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
```

The script will:
1. Load and preprocess the QA dataset
2. Fine-tune BERT for question answering
3. Evaluate on the test set

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

## License

See [LICENSE](LICENSE) for details.
