"""Абляция модальности: сколько информации с изображения доходит до LM.

Прогоняет ОДИН обученный чекпоинт на тесте в 4 режимах картинки
(real / blank / noise / shuffle) и печатает сравнительную таблицу.
Разрыв real − blank/shuffle = вклад изображения.

Usage:
    python -m graphvis_lp.scripts.ablation_eval /path/to/checkpoint
"""

from __future__ import annotations

import logging
import os
import sys

from datasets import load_from_disk
from peft import PeftModel

from graphvis_lp.config import TrainConfig
from graphvis_lp.training.evaluate import evaluate_link_prediction
from graphvis_lp.training.model import build_model_and_processor

logger = logging.getLogger(__name__)

MODES = ["real", "blank", "noise", "shuffle"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = TrainConfig()

    # База БЕЗ свежего LoRA — затем накатываем обученный адаптер.
    cfg.use_lora = False
    model, processor, tokenizer = build_model_and_processor(cfg)
    if ckpt:
        logger.info("Загрузка обученного адаптера: %s", ckpt)
        model = PeftModel.from_pretrained(model, ckpt)
    else:
        logger.warning("Чекпоинт не указан — оценивается БАЗОВАЯ модель (zero-shot).")

    test = load_from_disk(os.path.join(cfg.dataset_root, "stage2_test"))

    results = {}
    for mode in MODES:
        logger.info("=== image_mode=%s ===", mode)
        results[mode] = evaluate_link_prediction(
            model, processor, tokenizer, test, cfg, image_mode=mode,
        )

    # ── Сравнительная таблица ──
    print("\n" + "=" * 60)
    print(f"{'mode':<9} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>9}")
    print("-" * 60)
    for m in MODES:
        r = results[m]
        print(f"{m:<9} {r['MRR']:>8.4f} {r['Hits@1']:>7.1f}% {r['Hits@3']:>7.1f}% {r['Hits@10']:>8.1f}%")
    print("=" * 60)
    base = results["real"]["MRR"]
    for m in ("blank", "noise", "shuffle"):
        delta = base - results[m]["MRR"]
        share = 100 * delta / base if base else 0.0
        print(f"вклад изображения (real − {m}): ΔMRR={delta:+.4f}  ({share:+.0f}% от real)")
    print("Если real ≫ blank/shuffle — информация с картинки доходит до LM.")


if __name__ == "__main__":
    main()
