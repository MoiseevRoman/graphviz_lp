"""CLI: curriculum training (Stage 1 → Stage 2) + evaluation.

Usage:
    python -m graphvis_lp.scripts.train_and_eval
"""

from __future__ import annotations

import logging
import os
import random

from datasets import load_from_disk

from graphvis_lp.config import TrainConfig
from graphvis_lp.training.dataset import GraphVisLVLMDataset
from graphvis_lp.training.evaluate import evaluate_link_prediction
from graphvis_lp.training.model import build_model_and_processor
from graphvis_lp.training.trainer import make_trainer

logger = logging.getLogger(__name__)


def _load_hf_datasets(cfg: TrainConfig) -> dict:
    datasets = {}
    for name in ("stage1_graph_comprehension", "stage2_train", "stage2_valid", "stage2_test"):
        path = os.path.join(cfg.dataset_root, name)
        if os.path.exists(path):
            key = "stage1" if "stage1" in name else name
            datasets[key] = load_from_disk(path)
            logger.info("Загружен %s: %d примеров", key, len(datasets[key]))
        else:
            logger.warning("Не найден: %s", path)
    return datasets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = TrainConfig()
    random.seed(42)

    # Модель
    logger.info("Сборка модели …")
    model, processor, tokenizer = build_model_and_processor(cfg)

    # Данные
    logger.info("Загрузка датасетов …")
    raw = _load_hf_datasets(cfg)

    # ── Stage 1 ──────────────────────────────────────────────────
    if "stage1" in raw:
        s1_raw = raw["stage1"]
        if cfg.stage1_max_samples and len(s1_raw) > cfg.stage1_max_samples:
            s1_raw = s1_raw.select(random.sample(range(len(s1_raw)), cfg.stage1_max_samples))

        s1_ds = GraphVisLVLMDataset(s1_raw, processor, tokenizer, cfg)
        trainer_s1 = make_trainer(model, s1_ds, None, cfg, "stage1", tokenizer)
        logger.info("Stage 1 training …")
        trainer_s1.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
        trainer_s1.save_model()

    # ── Stage 2 ──────────────────────────────────────────────────
    if "stage2_train" in raw:
        s2_train_raw = raw["stage2_train"]
        if cfg.stage2_max_samples and len(s2_train_raw) > cfg.stage2_max_samples:
            s2_train_raw = s2_train_raw.select(random.sample(range(len(s2_train_raw)), cfg.stage2_max_samples))

        s2_train = GraphVisLVLMDataset(s2_train_raw, processor, tokenizer, cfg)
        s2_valid = (
            GraphVisLVLMDataset(raw["stage2_valid"], processor, tokenizer, cfg)
            if "stage2_valid" in raw else None
        )

        trainer_s2 = make_trainer(model, s2_train, s2_valid, cfg, "stage2", tokenizer)
        logger.info("Stage 2 training …")
        trainer_s2.train()
        trainer_s2.save_model()

    # ── Evaluation ───────────────────────────────────────────────
    if "stage2_test" in raw:
        logger.info("Evaluation …")
        evaluate_link_prediction(model, processor, tokenizer, raw["stage2_test"], cfg)

    logger.info("Готово.")


if __name__ == "__main__":
    main()