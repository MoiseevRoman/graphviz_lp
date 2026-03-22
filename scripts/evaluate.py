"""Evaluation: batched beam-search link prediction с метриками MRR / Hits@k."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from graphvis_lp.config import TrainConfig
from graphvis_lp.utils import normalize_answer

logger = logging.getLogger(__name__)

_FALLBACK_SIZE = (336, 336)
_FALLBACK_COLOR = (128, 128, 128)


def evaluate_link_prediction(
    model,
    processor,
    tokenizer,
    test_dataset,
    cfg: TrainConfig,
    num_beams: int | None = None,
    max_new_tokens: int | None = None,
) -> Dict[str, Any]:
    """Batched beam-search evaluation на тестовом split.

    Returns:
        Словарь с метриками: ``MRR, Hits@1, Hits@3, Hits@10, …``
    """
    num_beams = num_beams or cfg.eval_num_beams
    max_new_tokens = max_new_tokens or cfg.eval_max_new_tokens
    batch_size = cfg.eval_batch_size
    num_workers = cfg.eval_image_workers

    n_total = len(test_dataset)
    max_eval = min(cfg.eval_max_samples, n_total)

    random.seed(42)
    eval_indices = sorted(random.sample(range(n_total), max_eval)) if n_total > max_eval else list(range(n_total))

    device = next(model.parameters()).device
    image_token = getattr(processor, "image_token", "<image>")
    model.eval()

    logger.info(
        "EVALUATION: %d/%d samples, beams=%d, batch=%d",
        max_eval, n_total, num_beams, batch_size,
    )

    # ── загрузка одного сэмпла ───────────────────────────────────
    def _load_sample(data_i: int) -> Dict[str, Any]:
        sample = test_dataset[data_i]
        raw_path = sample[cfg.image_column].replace("\\", "/")
        full_path = os.path.join(cfg.image_root, raw_path) if cfg.image_root else raw_path

        try:
            img = Image.open(full_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", _FALLBACK_SIZE, _FALLBACK_COLOR)

        prompt = sample[cfg.prompt_column].strip()
        input_text = prompt if image_token in prompt else f"{image_token}\n{prompt}"

        return {
            "image": img,
            "input_text": input_text,
            "correct": sample[cfg.answer_column].strip(),
            "data_i": data_i,
        }

    # ── основной цикл ────────────────────────────────────────────
    ranks: List[float] = []
    detailed: List[Dict[str, Any]] = []
    found_count = 0
    t0 = time.time()

    for batch_start in range(0, max_eval, batch_size):
        batch_end = min(batch_start + batch_size, max_eval)
        batch_idx = eval_indices[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            samples = list(pool.map(_load_sample, batch_idx))

        inputs = processor(
            text=[s["input_text"] for s in samples],
            images=[s["image"] for s in samples],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                early_stopping=True,
            )

        for i, sample in enumerate(samples):
            correct_norm = normalize_answer(sample["correct"])
            beam_start = i * num_beams
            beam_end = (i + 1) * num_beams
            beam_seqs = output_ids[beam_start:beam_end]

            candidates, seen = [], set()
            for seq in beam_seqs:
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                norm = normalize_answer(text)
                if norm and norm not in seen:
                    seen.add(norm)
                    candidates.append(text)

            rank = float("inf")
            for r_idx, cand in enumerate(candidates, 1):
                if normalize_answer(cand) == correct_norm:
                    rank = r_idx
                    found_count += 1
                    break

            ranks.append(rank)
            detailed.append({
                "idx": sample["data_i"],
                "correct": sample["correct"],
                "rank": rank if rank != float("inf") else -1,
                "found": rank != float("inf"),
                "top5": candidates[:5],
            })

        processed = batch_end
        if processed % 50 < batch_size or processed == max_eval:
            elapsed = time.time() - t0
            speed = processed / max(elapsed, 1e-6)
            eta_min = (max_eval - processed) / max(speed, 0.01) / 60
            logger.info(
                "[%5d/%d] %.1f s/s | found=%d/%d | ETA %.0fm",
                processed, max_eval, speed, found_count, processed, eta_min,
            )

        if processed % 100 < batch_size:
            torch.cuda.empty_cache()

    # ── метрики ──────────────────────────────────────────────────
    r = np.array(ranks, dtype=float)
    reciprocal = np.where(np.isfinite(r), 1.0 / r, 0.0)
    finite = r[np.isfinite(r)]

    metrics: Dict[str, Any] = {
        "MRR": float(np.mean(reciprocal)),
        "Hits@1": float(np.mean(r <= 1)) * 100,
        "Hits@3": float(np.mean(r <= 3)) * 100,
        "Hits@10": float(np.mean(r <= 10)) * 100,
        "Found_Rate": float(found_count / max_eval) * 100,
        "Mean_Rank_found": float(np.mean(finite)) if len(finite) else float("inf"),
        "Median_Rank_found": float(np.median(finite)) if len(finite) else float("inf"),
        "n_evaluated": max_eval,
        "n_total_test": n_total,
    }

    total_min = (time.time() - t0) / 60
    logger.info("=" * 65)
    logger.info("RESULTS (%d queries, %.1f min)", max_eval, total_min)
    logger.info("  MRR        : %.4f", metrics["MRR"])
    logger.info("  Hits@1     : %.1f%%", metrics["Hits@1"])
    logger.info("  Hits@3     : %.1f%%", metrics["Hits@3"])
    logger.info("  Hits@10    : %.1f%%", metrics["Hits@10"])
    logger.info("  Found Rate : %.1f%%", metrics["Found_Rate"])
    logger.info("=" * 65)

    # Сохранение
    os.makedirs(cfg.output_dir, exist_ok=True)
    for fname, data in [("test_metrics.json", metrics), ("test_details.json", detailed)]:
        path = os.path.join(cfg.output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Сохранено: %s", path)

    return metrics