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


def quant_config(cfg: TrainConfig):
    """NF4-конфиг QLoRA; при cfg.vision_bf16 зрительная башня не квантуется."""
    if not cfg.use_4bit:
        return None
    kw: Dict[str, Any] = {}
    if getattr(cfg, "vision_bf16", False):
        # имя сегмента "visual" (Qwen2-VL / 2.5-VL: model.visual.*); lm_head — дефолт bnb
        kw["llm_int8_skip_modules"] = ["visual", "model.visual", "lm_head"]
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
        **kw,
    )


def vision_precision(model) -> Tuple[int, int]:
    """(квантованных 4-bit Linear, всего Linear) в зрительной башне."""
    n4 = n = 0
    for name, m in model.named_modules():
        if "visual" not in name.split("."):
            continue
        t = type(m).__name__
        if t in ("Linear4bit", "Linear8bitLt"):
            n4 += 1
            n += 1
        elif isinstance(m, torch.nn.Linear):
            n += 1
    return n4, n


def build_model_and_processor(
    cfg: TrainConfig,
) -> Tuple[Any, Any, PreTrainedTokenizerBase]:
    """Возвращает ``(model, processor, tokenizer)`` для LLaVA или Qwen2.5-VL."""

    family = detect_family(cfg.model_name, cfg.model_family)
    logger.info("Семейство модели: %s (%s)", family, cfg.model_name)

    load_kwargs: Dict[str, Any] = {"device_map": "cuda:0"}

    # Quantization
    qc = quant_config(cfg)
    if qc is not None:
        load_kwargs["quantization_config"] = qc
        logger.info("4-bit quantization (QLoRA) включена%s",
                    " (зрительная башня — bf16)" if getattr(cfg, "vision_bf16", False) else "")

    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16

    model, processor = load_model_and_processor(cfg, load_kwargs, dtype)
    if getattr(cfg, "vision_bf16", False):
        n4, n = vision_precision(model)
        logger.info("vision_bf16: квантованных Linear в зрительной башне %d из %d", n4, n)
        if n4:
            raise RuntimeError(f"vision_bf16: {n4}/{n} Linear зрительной башни всё же в 4 бит "
                               "(llm_int8_skip_modules не сработал в этой версии transformers)")
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
