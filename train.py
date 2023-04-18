import functools
from dataclasses import dataclass

import bitsandbytes as bnb
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from einops import rearrange
from transformers import DefaultDataCollator

from utils import get_model_and_tokenizer, save_checkpoint, tokenize_inputs


@dataclass
class Config:
    project_name: str
    model_name: str
    load_in_8bit: bool
    adapter_size: int
    max_seq_len: int
    adapter_dropout: float
    dataset: str
    batch_size: int
    microbatch_size: int
    max_lr: float
    num_epochs: int
    print_every: int
    save_every: int


def train(config: Config):
    model, tokenizer = get_model_and_tokenizer(
        config.model_name,
        config.adapter_size,
        config.max_seq_len,
        config.load_in_8bit,
        config.adapter_dropout,
    )

    dataset = load_dataset(config.dataset, split="train")
    tokenize_partial = functools.partial(
        tokenize_inputs,
        tokenizer=tokenizer,
        max_length=config.max_seq_len,
        adapter_size=config.adapter_size,
    )
    dataset = dataset.map(lambda examples: tokenize_partial(examples), batched=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=DefaultDataCollator(),
        batch_size=config.microbatch_size,
        pin_memory=True,
    )

    assert (
        config.batch_size % config.microbatch_size == 0
    ), "batch size must be divisible by microbatch size"
    accum_iters = int(config.batch_size // config.microbatch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(project=config.project_name, config=config.__dict__)
    run_name = run.name

    # Model, optimizer, scheduler, loss, loss scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        total_steps=config.num_epochs
        * (
            len(dataloader) + 10
        ),  # just to make sure scheduler doesn't run out if some off-by-one errors/restarts
        pct_start=0.05,
        div_factor=1000,
        final_div_factor=25,
        anneal_strategy="linear",
        three_phase=False,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    total_steps = 0
    running_loss = 0
    batch_loss = 0
    micro_batches = 0
    for epoch in range(config.num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)  # bsz, seq_len

            with torch.cuda.amp.autocast():
                outputs = model(input_ids)
                outputs = rearrange(
                    outputs, "b l v -> (b l) v"
                )  # bsz, seq_len, vocab_size
                micro_batch_loss = criterion(outputs, labels.view(-1))

            normalized_loss = micro_batch_loss / accum_iters
            batch_loss += normalized_loss.item()
            running_loss += normalized_loss.item()
            scaler.scale(normalized_loss).backward()
            micro_batches += 1

            if micro_batches == accum_iters:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                scaler.step(optimizer)
                scaler.update()

                wandb.log({"train_loss": batch_loss, "lr": scheduler.get_last_lr()[0]})
                total_steps += 1

                if (total_steps + 1) % config.print_every == 0:
                    print(
                        f"Step {total_steps + 1}: Loss = {running_loss / config.print_every:.4f}"
                    )
                    running_loss = 0

                if (total_steps) % config.save_every == 0:
                    print("Saving adapter weights...")
                    save_checkpoint(model, total_steps + 1, run_name)

                micro_batches = 0
                batch_loss = 0

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
