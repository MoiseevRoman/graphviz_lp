from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    PreTrainedTokenizerBase,
)

from graphvis_lp.config import TrainConfig

logger = logging.getLogger(__name__)


def build_model_and_processor(
    cfg: TrainConfig,
) -> Tuple[LlavaForConditionalGeneration, Any, PreTrainedTokenizerBase]:
    """Возвращает ``(model, processor, tokenizer)``."""

    load_kwargs: Dict[str, Any] = {"device_map": "cuda:0"}

    # Quantization
    if cfg.use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("4-bit quantization (QLoRA) включена")

    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16

    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        cache_dir=cfg.model_cache_dir,
        **load_kwargs,
    )
    processor = AutoProcessor.from_pretrained(cfg.model_name, cache_dir=cfg.model_cache_dir)
    tokenizer = (
        processor.tokenizer
        if hasattr(processor, "tokenizer")
        else AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.model_cache_dir)
    )

    # Специальные токены
    image_token = getattr(processor, "image_token", "<image>")
    extra = tokenizer.special_tokens_map_extended.get("additional_special_tokens", [])
    if image_token not in extra and image_token not in tokenizer.get_vocab():
        extra.append(image_token)
        tokenizer.add_special_tokens({"additional_special_tokens": extra})
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA
    if cfg.use_lora:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.lora_target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Gradient checkpointing (после LoRA)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing включён (use_reentrant=False)")

    return model, processor, tokenizer