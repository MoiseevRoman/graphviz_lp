"""Абстракция семейства моделей: изолирует ВСЕ отличия LLaVA vs Qwen2.5-VL / Qwen3-VL.

Смена бэкенда сводится к смене ``cfg.model_name`` — весь остальной код
(dataset / evaluate / probes / trainer) работает через функции ниже и не
знает, какая именно модель под капотом.

Отличия, которые здесь инкапсулированы:
    • класс модели и способ загрузки процессора;
    • формат промпта (спец-токен <image> у LLaVA vs chat-template у Qwen);
    • на выходе процессора: LLaVA даёт pixel_values фикс. формы (C,H,W),
      Qwen — плоские патчи (N_patch, dim) + image_grid_thw (n_img,3);
    • как следствие — РАЗНАЯ коллация батча (stack vs cat + grid);
    • какие модули оборачивать LoRA;
    • как найти позиции визуальных токенов в input_ids (для attention-проб).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Определение семейства
# ──────────────────────────────────────────────────────────────────
def detect_family(model_name: str, override: str = "auto") -> str:
    """Возвращает ``"llava"`` или ``"qwen"``."""
    if override and override != "auto":
        return override.lower()
    name = model_name.lower()
    if "qwen" in name and "vl" in name:
        return "qwen"
    if "llava" in name:
        return "llava"
    # По умолчанию — LLaVA-совместимый путь.
    return "llava"


def is_qwen3(model_name: str) -> bool:
    """Qwen3-VL идёт по пути семейства "qwen"; отличаются класс и шаг патча."""
    return "qwen3" in model_name.lower()


# ──────────────────────────────────────────────────────────────────
# Загрузка модели + процессора
# ──────────────────────────────────────────────────────────────────
def load_model_and_processor(cfg, load_kwargs: Dict[str, Any], dtype):
    """Грузит модель нужного класса и процессор. Возвращает (model, processor)."""
    family = detect_family(cfg.model_name, cfg.model_family)

    if cfg.attn_implementation:
        load_kwargs = {**load_kwargs, "attn_implementation": cfg.attn_implementation}

    if family == "qwen":
        from transformers import AutoProcessor

        if is_qwen3(cfg.model_name):
            # Qwen3-VL: тот же chat-template / <|image_pad|> / image_grid_thw, что у
            # Qwen2.5-VL, но патч 16 px × merge 2 = 32 px на токен (а не 28).
            try:
                from transformers import Qwen3VLForConditionalGeneration as _Cls
            except ImportError as e:
                raise ImportError("Qwen3-VL требует transformers>=4.57") from e
            min_px, max_px = cfg.qwen3_min_pixels, cfg.qwen3_max_pixels
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Cls
            min_px, max_px = cfg.qwen_min_pixels, cfg.qwen_max_pixels

        model = _Cls.from_pretrained(
            cfg.model_name, torch_dtype=dtype, cache_dir=cfg.model_cache_dir, **load_kwargs,
        )
        # min/max_pixels задаём процессору — так контролируем число визуальных токенов.
        processor = AutoProcessor.from_pretrained(
            cfg.model_name, cache_dir=cfg.model_cache_dir,
            min_pixels=min_px, max_pixels=max_px,
        )
    else:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            cfg.model_name, torch_dtype=dtype, cache_dir=cfg.model_cache_dir, **load_kwargs,
        )
        processor = AutoProcessor.from_pretrained(cfg.model_name, cache_dir=cfg.model_cache_dir)

    return model, processor


def lora_targets(cfg) -> List[str]:
    family = detect_family(cfg.model_name, cfg.model_family)
    return cfg.lora_target_modules_qwen if family == "qwen" else cfg.lora_target_modules


def image_token_str(processor, family: str) -> str:
    """Строковый плейсхолдер изображения для данного семейства."""
    if family == "qwen":
        return "<|image_pad|>"
    return getattr(processor, "image_token", "<image>")


def image_token_id(model, tokenizer, family: str) -> int:
    """ID токена-изображения (для локализации визуальных позиций)."""
    cfg = getattr(model, "config", None)
    if family == "qwen":
        tid = getattr(cfg, "image_token_id", None)
        if tid is None:
            tid = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        return int(tid)
    # LLaVA: все 576 визуальных токенов имеют один и тот же id.
    tid = getattr(cfg, "image_token_index", None)
    if tid is None:
        tid = tokenizer.convert_tokens_to_ids("<image>")
    return int(tid)


# ──────────────────────────────────────────────────────────────────
# Формирование текста промпта / полного примера
# ──────────────────────────────────────────────────────────────────
def format_eval_text(processor, family: str, prompt: str) -> str:
    """Текст запроса для ГЕНЕРАЦИИ (add_generation_prompt=True)."""
    prompt = prompt.strip()
    if family == "qwen":
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    tok = image_token_str(processor, family)
    return prompt if tok in prompt else f"{tok}\n{prompt}"


def build_train_texts(
    processor, tokenizer, family: str, prompt: str, answer: str,
) -> Tuple[str, str]:
    """Возвращает (prompt_text, full_text) для teacher-forcing.

    prompt_text — только запрос (для маскирования лейблов),
    full_text — запрос + ответ + eos.
    """
    prompt = prompt.strip()
    answer = answer.strip()
    prompt_text = format_eval_text(processor, family, prompt)

    if family == "qwen":
        eos = tokenizer.eos_token or "<|im_end|>"
        full_text = f"{prompt_text}{answer}{eos}"
    else:
        eos = tokenizer.eos_token or ""
        full_text = f"{prompt_text} {answer}{eos}"
    return prompt_text, full_text


# ──────────────────────────────────────────────────────────────────
# Процессинг одного примера (для Dataset.__getitem__)
# ──────────────────────────────────────────────────────────────────
def process_train_example(
    processor, tokenizer, family: str, prompt: str, answer: str, image, max_length: int,
) -> Dict[str, torch.Tensor]:
    """Строит (input_ids, attention_mask, labels, pixel_values[, image_grid_thw])."""
    prompt_text, full_text = build_train_texts(processor, tokenizer, family, prompt, answer)

    full = processor(
        text=full_text, images=image, return_tensors="pt",
        padding=False, truncation=True, max_length=max_length,
    )
    prompt_enc = processor(
        text=prompt_text, images=image, return_tensors="pt",
        padding=False, truncation=True, max_length=max_length,
    )
    input_ids = full["input_ids"][0]
    attention_mask = full["attention_mask"][0]
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Защита: если truncation обрезал ответ, лейблы вышли бы полностью
    # маскированными (пример не учит хвост). На текущем конфиге не срабатывает.
    prompt_len = min(prompt_len, input_ids.size(0))
    if prompt_len >= input_ids.size(0):
        logger.warning(
            "example truncated: answer fully cut (prompt_len=%d >= seq_len=%d)",
            prompt_len, input_ids.size(0),
        )

    labels = input_ids.clone()
    labels[:prompt_len] = -100

    out: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": full["pixel_values"][0] if family != "qwen" else full["pixel_values"],
    }
    if family == "qwen":
        # pixel_values Qwen: (N_patch, dim) — НЕ индексируем [0].
        out["pixel_values"] = full["pixel_values"]
        out["image_grid_thw"] = full["image_grid_thw"]  # (1, 3)
    return out


# ──────────────────────────────────────────────────────────────────
# Коллация батча
# ──────────────────────────────────────────────────────────────────
def collate(
    family: str, tokenizer, features: List[Dict[str, Any]], pad_to_multiple_of: Optional[int],
) -> Dict[str, torch.Tensor]:
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels_list = [f["labels"] for f in features]

    max_len = max(ids.size(0) for ids in input_ids)
    if pad_to_multiple_of:
        m = pad_to_multiple_of
        max_len = (max_len + m - 1) // m * m

    pad_id = tokenizer.pad_token_id

    def _pad(t: torch.Tensor, fill: int) -> torch.Tensor:
        n = max_len - t.size(0)
        return torch.cat([t, t.new_full((n,), fill)]) if n > 0 else t

    batch: Dict[str, torch.Tensor] = {
        "input_ids": torch.stack([_pad(ids, pad_id) for ids in input_ids]),
        "attention_mask": torch.stack([_pad(a, 0) for a in attention_mask]),
        "labels": torch.stack([_pad(lb, -100) for lb in labels_list]),
    }

    if family == "qwen":
        # Патчи всех картинок склеиваются по dim=0; grid_thw — по строкам.
        batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)
    else:
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])

    return batch


# ──────────────────────────────────────────────────────────────────
# Локализация визуальных токенов (для attention-проб)
# ──────────────────────────────────────────────────────────────────
def image_position_mask(input_ids: torch.Tensor, img_token_id: int) -> torch.Tensor:
    """Булева маска (той же формы, что input_ids) визуальных позиций."""
    return input_ids == img_token_id
