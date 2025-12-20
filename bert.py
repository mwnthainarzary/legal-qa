import json
import logging
import os
import torch
from datasets import Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer

# Disable tokenizer parallelism to avoid fork warnings with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 for faster matrix operations on Ampere GPUs (A6000)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Read Train Data
with open(r"data/train.json", "r", encoding="utf8") as read_file:
    train = json.load(read_file)

# Read Test Data
with open(r"data/test.json", "r", encoding="utf8") as read_file:
    test = json.load(read_file)


def flatten_qa_dataset(dataset):
    contexts = []
    questions = []
    answers = []

    for group in dataset:          # group is a list
        for doc in group:          # doc is a dict
            context = doc["context"]
            for qa in doc["questions"]:
                contexts.append(context)
                questions.append(qa["question"])
                answers.append({
                    "text": qa["answer"]["text"],
                    "answer_start": qa["answer"]["answer_start"]
                })

    return {
        "context": contexts,
        "question": questions,
        "answers": answers
    }



train_data = flatten_qa_dataset(train)
test_data  = flatten_qa_dataset(test)

train_dataset = Dataset.from_dict(train_data)
eval_dataset  = Dataset.from_dict(test_data)



tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def preprocess_qa(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        sequence_ids = tokenized.sequence_ids(i)

        start_token = end_token = 0
        for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
            if seq_id != 1:
                continue
            if offset[0] <= start_char < offset[1]:
                start_token = idx
            if offset[0] < end_char <= offset[1]:
                end_token = idx

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    tokenized.pop("offset_mapping")
    return tokenized

train_dataset = train_dataset.map(preprocess_qa, batched=True)
eval_dataset  = eval_dataset.map(preprocess_qa, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)

eval_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)



model = BertForQuestionAnswering.from_pretrained("bert-base-cased")

training_args = TrainingArguments(
    output_dir="./bert_qa_results",
    overwrite_output_dir=True,
    num_train_epochs=25,
    per_device_train_batch_size=128,  # Push batch size higher
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_strategy="no",
    report_to="none",
    bf16=True,                        # BF16 is better on Ampere GPUs than FP16
    dataloader_num_workers=8,         # More parallel data loading
    dataloader_pin_memory=True,       # Faster CPU->GPU transfer
    dataloader_prefetch_factor=4,     # Prefetch more batches
    remove_unused_columns=False,      # Keep all columns for QA model
    optim="adamw_torch_fused",        # Fused optimizer is faster
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer
)

trainer.train()
trainer.evaluate()
