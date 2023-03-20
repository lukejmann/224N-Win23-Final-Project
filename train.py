import argparse

import bitsandbytes as bnb
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import (
    DataCollatorForLanguageModeling,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)

from model_config import load_model_and_tokenizer
from prompts import generate_prompt

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument(
    "--model_type", default="LoRA", help="Model type: base, LoRA, Pre, or PreLoRA"
)
parser.add_argument("--subject", default="base", help="Biden or Trump")
parser.add_argument(
    "--from_epoch", default=None, help="(Optional) Epoch to start training from"
)
parser.add_argument(
    "--n_epochs", default=1, help="(Optional) Number of epochs to train for"
)
args = parser.parse_args()

# subject is required for any model other than base
if args.model_type.lower() != "base" and args.subject.lower() not in ["biden", "trump"]:
    raise ValueError("Subject must be specified for any model other than base")


# Load model and tokenizer
base_model, tokenizer = load_model_and_tokenizer("base", args.subject, args.from_epoch)
if args.model_type.lower() == "lora":
    base_model = prepare_model_for_int8_training(base_model)
    tokenizer.pad_token_id = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


num_learnable_tokens = 1
cutoff_len = 512
pre_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=num_learnable_tokens,
    encoder_hidden_size=512,
    prefix_projection=True,
)


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

    # Add learnable tokens to input_ids and attention_mask
    num_virtual_tokens = pre_config.num_virtual_tokens
    input_ids = [num_virtual_tokens] + input_ids
    attention_mask = [1] + attention_mask

    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attention_mask, device=device),
    }


val_size = 1000
data = load_dataset(
    "json",
    data_files=f"data/finalized_data/{args.subject.lower()}/{args.subject.lower()}_train.json",
)
train_val = data["train"].train_test_split(test_size=val_size, shuffle=False, seed=224)
train_data = train_val["train"]
val_data = train_val["test"]

if args.model_type.lower() == "pre-tune":
    train_data = train_data.shuffle().map(
        lambda x: tokenize_prefix_tuning(generate_prompt(x, args.subject))
    )
    val_data = val_data.shuffle().map(
        lambda x: tokenize_prefix_tuning(generate_prompt(x, args.subject), args.subject)
    )
else:
    train_data = train_data.shuffle().map(
        lambda x: tokenize(generate_prompt(x, args.subject))
    )
    val_data = val_data.shuffle().map(
        lambda x: tokenize(generate_prompt(x, args.subject))
    )


if args.from_epoch is not None:
    from_epoch = int(args.from_epoch)
else:
    from_epoch = 0

if args.n_epochs is not None:
    n_epochs = int(args.n_epochs)
else:
    n_epochs = 1


def train_lora():
    print(
        f"Training LoRA model for subject {args.subject.lower()} for {n_epochs} epochs"
    )

    model = get_peft_model(base_model, lora_config)

    trainer = Trainer(
        model=base_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            run_name=f"llama-7b-{args.model_type.lower()}",
            gradient_accumulation_steps=128 // 1,
            # max_steps=1024,
            warmup_steps=12,
            num_train_epochs=n_epochs,
            learning_rate=3e-4,
            # fp16=True,
            logging_steps=25,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=25,
            save_steps=25,
            output_dir=f"models/{args.subject.lower()}/{args.model_type.lower()}",
            save_total_limit=5,
            load_best_model_at_end=True,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train()
    save_path = f"models/{args.subject.lower()}/{args.model_type.lower()}{n_epochs}"
    model.save_pretrained(save_path)
    print("Model saved")


def train_prefix_tuning():
    print(
        f"Training Prefix Tuning model for subject {args.subject.lower()} for {n_epochs} epochs"
    )

    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.enable_input_require_grads()

    trainer = Trainer(
        model=base_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            run_name=f"llama-7b-{args.model_type.lower()}",
            max_steps=52,
            warmup_steps=2,
            num_train_epochs=n_epochs,
            learning_rate=3e-4,
            logging_steps=25,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=25,
            save_steps=25,
            output_dir=f"models/{args.subject.lower()}/{args.model_type.lower()}",
            save_total_limit=1,
            load_best_model_at_end=True,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # model.to(device)
    model.config.use_cache = False
    trainer.train()
    save_path = f"models/{args.subject.lower()}/{args.model_type.lower()}{n_epochs}"
    model.save_pretrained(save_path)
    print("Model saved to ", save_path)


def train_pre_lora():
    pass


if args.model_type.lower() == "base":
    # train_base()
    print("Training base model")
elif args.model_type.lower() == "lora":
    print("Training LoRA model")
    train_lora()
elif args.model_type.lower() == "pre-tune":
    print("Training Pre model")
    train_prefix_tuning()
