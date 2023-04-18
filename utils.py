import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapter import AdapterModel


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} |"
        + f"| all params: {all_param} |"
        + f"| trainable%: {100 * trainable_params / all_param}"
    )


def get_model_and_tokenizer(
    model_name: str,
    adapter_size: int,
    max_seq_len: int,
    load_in_8bit: bool = False,
    adapter_dropout: float = 0.1,
) -> torch.nn.Module:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # key to success!
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_8bit=load_in_8bit, device_map="auto"
    )
    pretrained_model.gradient_checkpointing_enable()  # i'm not actually sure if this is necessary

    model = AdapterModel(
        pretrained_model,
        max_len=max_seq_len,
        adapter_size=adapter_size,
        adapter_dropout=adapter_dropout,
    )
    print_trainable_parameters(model)

    return model, tokenizer


def tokenize_inputs(examples, tokenizer, max_length, adapter_size):
    max_input_length = max_length - adapter_size
    newline_token = tokenizer("\n", return_tensors="pt").input_ids.view(-1)  # (1, )
    out = {"labels": [], "input_ids": []}
    for i, (prompt, response) in enumerate(
        zip(examples["prompt"], examples["response"])
    ):
        # combine prompt and response into single sequence
        input_tokens = tokenizer(prompt, return_tensors="pt").input_ids.view(
            -1
        )  # (prompt_len, )
        target_tokens = tokenizer(response, return_tensors="pt")["input_ids"].view(
            -1
        )  # (response_len, )
        combined = torch.cat([input_tokens, newline_token, target_tokens])

        # get input ids
        input_ids = torch.full((max_input_length,), tokenizer.pad_token_id)
        input_ids[0 : min(combined.numel(), max_input_length)] = combined[
            :max_input_length
        ]  # if length is less than max_input_length, the first pad token will be treated as eos

        # labels-including padding for the adapter (inputs and labels will be different lengths!)
        labels = torch.full((max_length,), -100)
        ## the only thing we want to fill in is the target tokens.
        tgt_start = (
            input_tokens.numel() + adapter_size
        )  # + 1 for the newline, - 1 to shift left by 1
        tgt_end = min(max_length, tgt_start + target_tokens.numel())
        labels[tgt_start:tgt_end] = target_tokens[: tgt_end - tgt_start]
        if tgt_end < max_length:
            labels[tgt_end] = tokenizer.pad_token_id

        out["input_ids"].append(input_ids)
        out["labels"].append(labels)

    out = {k: torch.stack(v) for k, v in out.items()}
    return out


def save_checkpoint(model, steps, run):
    file_name = f"/storage/{run}-step-{steps}.npy"
    np.save(file_name, model.adapter.data.cpu().numpy())


def restore_from_checkpoint(checkpoint_path, model, dataloader, scheduler):
    # load the adapter weights into the model

    # fast-forward the dataloader

    # fast-forward the scheduler

    pass
