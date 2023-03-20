import glob
import os

import gradio as gr
import torch
import transformers
from peft import PeftModel

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


def get_model_path(subject, model_type, from_epoch):
    if model_type == "base":
        return f"decapoda-research/llama-7b-hf"
    model_paths = glob.glob(
        f"models/{subject}/{model_type}/checkpoint-{from_epoch}*"
        if from_epoch
        else f"models/{subject}/{model_type}/*"
    )
    model_paths = [p for p in model_paths if "hf" not in p]
    model_paths = sorted(model_paths, key=lambda x: os.path.getmtime(x))
    if len(model_paths) == 0:
        return None
    model_path = model_paths[-1]
    return model_path


def load_model_and_tokenizer(model_type, subject, from_epoch):
    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

    if model_type == "base":
        print(f"Loading model: {model_type} for {subject}")
        model = PeftModel.from_pretrained(
            model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16
        )

        return model, tokenizer

    if model_type == "pre-tune":
        print(f"Loading model: {model_type} for {subject}")
        return model, tokenizer

    print(f"Loading model: {model_type} for {subject}")
    model_path = get_model_path(subject, model_type, from_epoch)
    if model_path is None:
        model = PeftModel.from_pretrained(
            model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16
        )

        print(f"------WARNING: {model_path} not found, using base model instead-------")
        return model, tokenizer
    print(f"Loading model from {model_path}")
    model = PeftModel.from_pretrained(
        model,
        "models/biden/lora-3",
        torch_dtype=torch.float16,
        # device_map={"": 0},
    )
    return model, tokenizer
