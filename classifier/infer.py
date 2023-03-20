import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main():
    test_data = load_dataset(
        "json",
        data_files=f"dataset/trump.json",
    )["train"]
    inputs = tokenizer(
        test_data["text"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "trump_checkpoints/checkpoint-6500"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    print(logits)
    predicted_class_id = torch.argmax(logits, dim=-1)
    print(predicted_class_id)


if __name__ == "__main__":
    main()
