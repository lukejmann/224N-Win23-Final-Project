import os

import torch
import torch.nn as nn
import bitsandbytes as bnb

from datasets import load_dataset
import transformers
import time
import glob


def speaker_id_to_name(speaker_id):
    if speaker_id.lower() == "biden":
        return "Joe Biden"
    elif speaker_id.lower() == "trump":
        return "Donald Trump"
    else:
        raise ValueError(f"Invalid speaker id: {speaker_id}")


def generate_prompt(context, speaker_id):
    return f"""Below is context that describes a conversation, paired with the name of the next speaker in the conversation. Write a response that appropriately continues the conversation as the next speaker.

### Context:
{context}

### Speaker:
{speaker_id_to_name(speaker_id)}

### {speaker_id_to_name(speaker_id)}:"""


def create_context_and_generate_prompt(input, speaker):
    context = f"PersonA: '{input}'"
    return generate_prompt(context, speaker)
