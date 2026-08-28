from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from graphvis_lp.config import TrainConfig
from graphvis_lp.training.families import (
    detect_family,
    image_token_str,
    load_model_and_processor,
    lora_targets,
)

logger = logging.getLogger(__name__)


def build_model_and_processor(
    cfg: TrainConfig,
) -> Tuple[Any, Any, PreTrainedTokenizerBase]:
    """Возвращает ``(model, processor, tokenizer)`` для LLaVA или Qwen2.5-VL."""

    family = detect_family(cfg.model_name, cfg.model_family)
    logger.info("Семейство модели: %s (%s)", family, cfg.model_name)

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

    model, processor = load_model_and_processor(cfg, load_kwargs, dtype)
    tokenizer = (
        processor.tokenizer
        if hasattr(processor, "tokenizer")
        else AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=cfg.model_cache_dir)
    )

    # Спец-токен изображения: у LLaVA <image> может отсутствовать в словаре,
    # у Qwen (<|image_pad|>) он всегда есть → второй ветки не будет.
    image_token = image_token_str(processor, family)
    extra = tokenizer.special_tokens_map_extended.get("additional_special_tokens", [])
    if image_token not in extra and image_token not in tokenizer.get_vocab():
        extra.append(image_token)
        tokenizer.add_special_tokens({"additional_special_tokens": extra})
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA (таргеты зависят от семейства)
    if cfg.use_lora:
        targets = list(lora_targets(cfg))
        if getattr(cfg, "train_vision", False):
            import torch.nn as _nn

            def _is_linear(m):
                return isinstance(m, _nn.Linear) or type(m).__name__ in (
                    "Linear4bit", "Linear8bitLt")

            # Полные имена всех Linear в зрительной башне (сегмент пути == "visual"):
            # ViT-attention (qkv/proj), MLP блоков, merger. Передаём полными именами —
            # PEFT матчит точно, независимо от версии Qwen2-VL / 2.5-VL.
            vis = [n for n, m in model.named_modules()
                   if _is_linear(m) and "visual" in n.split(".")]
            targets = targets + vis
            logger.info("train_vision: +%d визуальных Linear в LoRA (ViT+merger)", len(vis))

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
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
