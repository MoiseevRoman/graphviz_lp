"""Стоимость подачи графа в токенах — ось «денег/компьюта» кривой масштабирования.

Ключевой момент из обсуждения: кривая «качество vs число объектов» бессмысленна
без второй оси — СКОЛЬКО это стоит по токенам. И тут модальности принципиально
разные:

    • LLaVA-1.5 — CLIP-336 с ФИКСИРОВАННОЙ сеткой: ВСЕГДА 576 визуальных токенов,
      сколько ни рисуй. Значит по оси «стоимость от числа объектов» у LLaVA
      плоская линия — кривую стоимости картинки на нём построить нельзя.
    • Qwen2.5-VL — native dynamic resolution: число визуальных токенов ≈ площади
      картинки после smart_resize. Стоимость РАСТЁТ с разрешением/наполнением →
      кривую масштабирования строим ТОЛЬКО на Qwen.
    • Текст — стоимость = число текстовых токенов промпта (растёт с числом рёбер).

Здесь: точная реализация Qwen ``smart_resize`` (та же, что в HF-процессоре),
подсчёт текстовых токенов и агрегация по датасету/примеру. ``smart_resize`` —
чистая арифметика, тестируется без модели.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

# patch=14, spatial_merge_size=2 → один визуальный токен = блок 28×28 px.
QWEN_FACTOR = 28
QWEN_TOKEN_PX = QWEN_FACTOR * QWEN_FACTOR   # 784 px² на визуальный токен
LLAVA_VISUAL_TOKENS = 576                   # CLIP-336/14 → 24×24, фиксировано


# ────────────────────────────────────────────────────────────────
# Qwen smart_resize (портирование из transformers)
# ────────────────────────────────────────────────────────────────
def qwen_smart_resize(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = QWEN_FACTOR,
) -> tuple[int, int]:
    """Возвращает (h_bar, w_bar) — размер, к которому Qwen приведёт картинку.

    Точная копия ``Qwen2VLImageProcessor.smart_resize``: округление стороны до
    кратной ``factor`` с масштабированием под окно [min_pixels, max_pixels].
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_visual_tokens(
    width: int,
    height: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = QWEN_FACTOR,
) -> int:
    """Число визуальных токенов Qwen для картинки width×height px."""
    h_bar, w_bar = qwen_smart_resize(height, width, min_pixels, max_pixels, factor)
    return (h_bar * w_bar) // QWEN_TOKEN_PX


def visual_tokens(
    family: str,
    width: int,
    height: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> int:
    """Визуальные токены по семейству модели. LLaVA — константа, Qwen — от пикселей."""
    if family == "qwen":
        if min_pixels is None or max_pixels is None:
            raise ValueError("Для Qwen нужны min_pixels/max_pixels")
        return qwen_visual_tokens(width, height, min_pixels, max_pixels)
    return LLAVA_VISUAL_TOKENS


# ────────────────────────────────────────────────────────────────
# Замер стоимости на реальном процессоре (для eval-метрик)
# ────────────────────────────────────────────────────────────────
def measure_token_cost(
    processor,
    tokenizer,
    dataset,
    cfg,
    prompt_key: str,
    image_mode: str = "real",
    n_sample: int = 64,
) -> Dict[str, float]:
    """Средняя стоимость примера в токенах: total / text / visual.

    Прогоняет процессор (БЕЗ генерации) на ``n_sample`` примерах в тех же
    условиях, что и eval (тот же prompt_key, тот же режим картинки), и считает:
        visual = визуальные токены (Qwen: из image_grid_thw; LLaVA: 576);
        text   = все остальные токены промпта (чистая текстовая стоимость);
        total  = длина input_ids.
    Не требует GPU/весов — только процессор.
    """
    import os
    import numpy as np
    from PIL import Image

    # Репозиторий мешает два корня импорта (bare и graphvis_lp.*) — поддержим оба.
    try:
        from graphvis_lp.training.families import (
            detect_family, format_eval_text, image_token_id,
        )
    except ImportError:
        from training.families import (
            detect_family, format_eval_text, image_token_id,
        )

    family = detect_family(cfg.model_name, cfg.model_family)
    n = min(n_sample, len(dataset))
    if n == 0:
        return {"mean_total_tokens": 0.0, "mean_text_tokens": 0.0, "mean_visual_tokens": 0.0}

    s = int(getattr(cfg, "image_size", 336))
    totals, texts, visuals = [], [], []
    img_id = None

    for i in range(n):
        sample = dataset[i]
        prompt = sample[prompt_key]
        if image_mode == "blank":
            img = Image.new("RGB", (s, s), (255, 255, 255))
        else:
            raw = sample[cfg.image_column].replace("\\", "/")
            path = os.path.join(cfg.image_root, raw) if cfg.image_root else raw
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (s, s), (128, 128, 128))

        text = format_eval_text(processor, family, prompt)
        enc = processor(text=[text], images=[img], return_tensors="pt")
        total = int(enc["input_ids"].shape[1])

        if family == "qwen" and "image_grid_thw" in enc:
            thw = enc["image_grid_thw"][0]
            vis = int(thw[0] * thw[1] * thw[2]) // 4    # merge 2×2
        else:
            if img_id is None:
                img_id = image_token_id(getattr(processor, "model", None) or processor,
                                        tokenizer, family)
            try:
                vis = int((enc["input_ids"][0] == img_id).sum())
            except Exception:
                vis = LLAVA_VISUAL_TOKENS

        totals.append(total)
        visuals.append(vis)
        texts.append(max(total - vis, 0))

    return {
        "mean_total_tokens": float(np.mean(totals)),
        "mean_text_tokens": float(np.mean(texts)),
        "mean_visual_tokens": float(np.mean(visuals)),
        "n_cost_sampled": n,
    }
