import argparse
import json
import os
import random

import nltk
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from model_config import load_model_and_tokenizer
from prompts import create_context_and_generate_prompt, speaker_id_to_name

parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument(
    "--model_type", default="base", help="Model type: base, LoRA, Pre, or PreLoRA"
)
parser.add_argument("--subject", default="biden", help="Subject: biden or trump")
parser.add_argument(
    "--n_samples", type=int, default=100, help="Number of samples to evaluate"
)
parser.add_argument(
    "--mode",
    choices=["eval", "user"],
    default="eval",
    help="Evaluation mode: eval or user",
)
parser.add_argument(
    "--user_input", type=str, help="User input for generating a response in user mode"
)
parser.add_argument("--at_epoch", default=None, help="Epoch to load model from")
args = parser.parse_args()

base_model_path = "decapoda-research/llama-7b-hf"

model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
if args.model_type.lower() == "lora":
    path = f"models/{args.subject.lower()}/lora-1"
    print(f"Loading model from {path}...")
    model = PeftModel.from_pretrained(
        model,
        "models/biden/lora",
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
elif args.model_type.lower() == "pre-tune":
    path = f"models/{args.subject.lower()}/pre1"
    print(f"Loading model from {path}...")
    model = PeftModel.from_pretrained(
        model,
        path,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )


tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
model.eval()


classifier_tokenizer = AutoTokenizer.from_pretrained(
    f"./classifier/{args.subject.lower()}_classifier/checkpoint-4550"
)

classifier_model = AutoModelForSequenceClassification.from_pretrained(
    f"./classifier/{args.subject.lower()}_classifier/checkpoint-4550"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_learnable_tokens = 1
cutoff_len = 512


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len + 1,
        padding="max_length",
    )
    return {
        "input_ids": torch.tensor(result["input_ids"][:-1], device=device),
        "attention_mask": torch.tensor(result["attention_mask"][:-1], device=device),
    }


def tokenize_prefix_tuning(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len + 1,
        padding="max_length",
    )

    input_ids = result["input_ids"][:-1]
    attention_mask = result["attention_mask"][:-1]

    num_virtual_tokens = num_virtual_tokens
    input_ids = [num_virtual_tokens] + input_ids
    attention_mask = [1] + attention_mask

    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attention_mask, device=device),
    }


def eval_method_classifier(generated_responses, actual_responses):
    scores = []
    for generated_response in generated_responses:
        inputs = tokenizer(generated_response, return_tensors="pt")
        with torch.no_grad():
            logits = classifier_model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            scores.append(predicted_class_id)
    avg_score = sum(scores) / len(scores)
    print(f"Average score: {avg_score:.4f}")
    return avg_score


def eval_method_blue(generated_responses, actual_responses):
    avg_bleu_score = nltk.translate.bleu_score.corpus_bleu(
        actual_responses, generated_responses
    )

    print(f"Average BLEU score: {avg_bleu_score:.4f}")
    return avg_bleu_score


def generate(input):
    prompt = create_context_and_generate_prompt(input, args.subject)
    inputs = (
        tokenize(prompt)
        if args.model_type.lower() != "pre-tune"
        else tokenize_prefix_tuning(prompt)
    )
    input_ids = inputs["input_ids"].cuda()
    print(input_ids.shape)
    generation_config = GenerationConfig(
        temperature=0.5, top_p=0.75, top_k=40, num_beams=1
    )

    model.config.use_cache = True
    print("Generating response...")
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=100,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split(f"### {speaker_id_to_name(args.subject)}:")[1].strip()


if args.mode == "eval":
    test_data = load_dataset(
        "json",
        data_files=f"data/finalized_data/{args.subject}/{args.subject}_test.json",
    )["train"]
    generated_responses = []
    actual_responses = []

    for i in range(args.n_samples):
        j = random.randint(0, len(test_data) - 1)
        user_input = test_data[j]["context"]
        actual_responses.append(test_data[j]["response"])
        generated_responses.append(generate(user_input))

    classifier_score = eval_method_classifier(generated_responses, actual_responses)
    avg_bleu_score = eval_method_blue(generated_responses, actual_responses)

    os.makedirs("./evals", exist_ok=True)
    current_model = args.at_epoch if args.at_epoch else "latest"
    with open(
        f"./evals/{args.subject}_{args.model_type}_{current_model}.json", "w"
    ) as f:
        json.dump(
            {
                "generated_responses": generated_responses,
                "actual_responses": actual_responses,
            },
            f,
        )

    print(f"Generated {args.n_samples} responses.")
    print("Evaluating with classifier...")
    print(f"Classifier score: {classifier_score:.4f}")
    print("Evaluating with BLEU...")
    print(f"BLEU score: {avg_bleu_score:.4f}")


elif args.mode == "user":
    user_input = args.user
    if not user_input:
        raise ValueError("User input is required for user mode.")
    generated_response = generate(user_input)
    # print(f"Generated Response: {generated_response}")
