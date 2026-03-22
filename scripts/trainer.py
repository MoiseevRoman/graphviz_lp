from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from graphvis_lp.config import TrainConfig
from graphvis_lp.training.dataset import DataCollatorForGraphVisLVLM

logger = logging.getLogger(__name__)


def make_trainer(
    model,
    train_dataset,
    eval_dataset: Optional[object],
    cfg: TrainConfig,
    run_name: str,
    tokenizer,
) -> Trainer:
    """Создаёт стандартный ``Trainer`` с нужными ``TrainingArguments``."""

    output_dir = os.path.join(cfg.output_dir, run_name)
    is_stage1 = "stage1" in run_name
    num_epochs = cfg.stage1_num_epochs if is_stage1 else cfg.stage2_num_epochs
    has_eval = eval_dataset is not None

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=cfg.eval_steps if has_eval else None,
        save_total_limit=cfg.save_total_limit,
        remove_unused_columns=False,
        bf16=cfg.use_bf16,
        fp16=not cfg.use_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        group_by_length=cfg.group_by_length,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["tensorboard"],
        logging_first_step=True,
    )

    collator = DataCollatorForGraphVisLVLM(tokenizer=tokenizer, pad_to_multiple_of=8)

    callbacks = []
    if has_eval:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Trainer создан: %s | epochs=%d | lr=%.1e | trainable=%s",
        run_name, num_epochs, cfg.learning_rate, f"{trainable:,}",
    )

    return trainer