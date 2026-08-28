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
from graphvis_lp.token_cost import measure_token_cost
from graphvis_lp.training.families import detect_family, format_eval_text
from graphvis_lp.utils import normalize_answer

logger = logging.getLogger(__name__)

_FALLBACK_COLOR = (128, 128, 128)


def _fb_size(cfg) -> tuple[int, int]:
    """Размер fallback/blank/noise-картинки = размер рендера датасета
    (cfg.image_size). Держим blank и real одинаковыми, чтобы у Qwen совпадало
    число визуальных токенов и абляция real−blank не загрязнялась."""
    s = int(getattr(cfg, "image_size", 336))
    return (s, s)


def evaluate_link_prediction(
    model,
    processor,
    tokenizer,
    test_dataset,
    cfg: TrainConfig,
    num_beams: int | None = None,
    max_new_tokens: int | None = None,
    image_mode: str = "real",
) -> Dict[str, Any]:
    """Batched beam-search evaluation на тестовом split.

    ``image_mode`` управляет абляцией модальности (вклад изображения):
        "real"    — настоящее изображение окрестности (по умолчанию);
        "blank"   — белый холст (картинка присутствует, но пустая);
        "noise"   — случайный шум (картинка бессмысленна);
        "shuffle" — изображение от ДРУГОГО запроса (структура не та).
    Разрыв метрик real − blank/shuffle = сколько информации с картинки
    реально доходит до языковой модели.

    ВНИМАНИЕ: MRR/Hits@k здесь — UNFILTERED ранги по позиции золотого в списке
    dedup'нутых beam-строк, ограниченные ``num_beams`` (не найден → reciprocal=0),
    с substring/normalize-матчем. Это НЕ filtered-протокол: нет исключения прочих
    истинных хвостов и нет общего набора кандидатов. Эти числа НЕ сопоставимы с
    filtered rank/sample-режимами (``evaluate_link_prediction_rank`` /
    ``…_sample``) и с таблицами KGE/KGT5 — не ставить их в одну таблицу.

    Returns:
        Словарь с метриками: ``MRR, Hits@1, Hits@3, Hits@10, …``
        (помечен ``eval_mode="generative_unfiltered"``).
    """
    num_beams = num_beams or cfg.eval_num_beams
    max_new_tokens = max_new_tokens or cfg.eval_max_new_tokens
    batch_size = cfg.eval_batch_size
    num_workers = cfg.eval_image_workers

    n_total = len(test_dataset)
    max_eval = min(cfg.eval_max_samples, n_total)

    random.seed(42)
    eval_indices = sorted(random.sample(range(n_total), max_eval)) if n_total > max_eval else list(range(n_total))

    # Для режима shuffle заранее строим соответствие data_i → путь ДРУГОГО
    # запроса (сдвиг на 1 по eval-набору), чтобы структура была валидной,
    # но не относящейся к текущему запросу.
    shuffle_lookup: Dict[int, str] = {}
    if image_mode == "shuffle":
        paths = [test_dataset[i][cfg.image_column] for i in eval_indices]
        rolled = paths[1:] + paths[:1]
        shuffle_lookup = dict(zip(eval_indices, rolled))

    device = next(model.parameters()).device
    family = detect_family(cfg.model_name, cfg.model_family)
    model.eval()

    # Батчевая beam-генерация у decoder-only требует ЛЕВОГО паддинга — иначе
    # правый паддинг сдвигает позиции и портит выход (HF предупреждает об этом).
    # Обучение использует правый паддинг (там нужны labels), поэтому ставим left
    # только на время оценки и восстанавливаем прежнее значение в конце.
    _pad_saved = getattr(tokenizer, "padding_side", None)
    for _tok in (tokenizer, getattr(processor, "tokenizer", None)):
        if _tok is not None:
            _tok.padding_side = "left"

    logger.info(
        "EVALUATION: %d/%d samples, beams=%d, batch=%d, image_mode=%s",
        max_eval, n_total, num_beams, batch_size, image_mode,
    )

    # ── загрузка одного сэмпла ───────────────────────────────────
    def _load_sample(data_i: int) -> Dict[str, Any]:
        sample = test_dataset[data_i]

        fb = _fb_size(cfg)
        if image_mode == "blank":
            img = Image.new("RGB", fb, (255, 255, 255))
        elif image_mode == "noise":
            arr = np.random.randint(0, 256, (*fb, 3), dtype=np.uint8)
            img = Image.fromarray(arr, "RGB")
        else:
            raw = shuffle_lookup[data_i] if image_mode == "shuffle" else sample[cfg.image_column]
            raw_path = raw.replace("\\", "/")
            full_path = os.path.join(cfg.image_root, raw_path) if cfg.image_root else raw_path
            try:
                img = Image.open(full_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", fb, _FALLBACK_COLOR)

        input_text = format_eval_text(processor, family, sample[cfg.prompt_column])

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
        "eval_mode": "generative_unfiltered",   # UNFILTERED, ранг по позиции beam-строки
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

    # Сохранение (суффикс режима, чтобы абляции не затирали друг друга)
    metrics["image_mode"] = image_mode
    suffix = "" if image_mode == "real" else f"_{image_mode}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    for fname, data in [(f"test_metrics{suffix}.json", metrics), (f"test_details{suffix}.json", detailed)]:
        path = os.path.join(cfg.output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Сохранено: %s", path)

    # Восстанавливаем прежний padding_side (обучение использует правый).
    if _pad_saved is not None:
        for _tok in (tokenizer, getattr(processor, "tokenizer", None)):
            if _tok is not None:
                _tok.padding_side = _pad_saved

    return metrics


def _load_eval_image(sample, cfg, image_mode, override_path=None):
    """Картинка для оценки (абляция модальности):
        real    — настоящая окрестность запроса;
        blank   — белый холст (картинка есть, но пустая);
        noise   — случайный шум (картинка бессмысленна);
        shuffle — картинка ОТ ДРУГОГО запроса (структура валидна, но не та).
    ``override_path`` задаёт путь для shuffle."""
    fb = _fb_size(cfg)
    if image_mode == "blank":
        return Image.new("RGB", fb, (255, 255, 255))
    if image_mode == "noise":
        arr = np.random.randint(0, 256, (*fb, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")
    raw = (override_path if override_path is not None else sample[cfg.image_column]).replace("\\", "/")
    full = os.path.join(cfg.image_root, raw) if cfg.image_root else raw
    try:
        return Image.open(full).convert("RGB")
    except Exception:
        return Image.new("RGB", fb, _FALLBACK_COLOR)


def evaluate_link_prediction_rank(
    model,
    processor,
    tokenizer,
    test_dataset,
    cfg: TrainConfig,
    candidate_provider,
    image_mode: str = "real",
    cand_batch: int = 8,
    max_length: int = 2048,
    realistic_ties: bool = False,
) -> Dict[str, Any]:
    """Filtered link prediction РАНЖИРОВАНИЕМ кандидатов по правдоподобию.

    ``realistic_ties``: при равенстве очков золотого с дистракторами ранг =
    1 + higher + ties/2 (realistic, как в pykeen и в sample-пути). По умолчанию
    False — сохраняется исторический optimistic тай-брейк (строгое «>»), чтобы
    ранее посчитанные rank-числа воспроизводились; для нового прогона включать
    флаг (см. --realistic_ties в viability_gate).

    Вместо открытой beam-генерации (медленно, ~beams×max_new_tokens шагов
    декодирования на батч, хрупкий substring-матч) скорим каждого
    type-constrained кандидата-хвоста teacher-forcing'ом с картинкой и ранжируем
    истинный хвост среди них — стандартный **filtered**-протокол KGC. Один forward
    на кандидата (батчами), без авторегрессии → на порядок быстрее и масштабируется
    на полный тест.

    ``candidate_provider(sample) -> (candidates, filter_others, gold)`` или None
    (пропустить пример). ``candidates`` — список имён-хвостов (включая ``gold`` в
    той же поверхностной форме, что в обучающих ответах); ``filter_others`` —
    прочие истинные хвосты (h,r,·), убираемые из ранжирования (filtered).
    """
    import numpy as _np

    family = detect_family(cfg.model_name, cfg.model_family)
    device = next(model.parameters()).device
    model.eval()
    eos = tokenizer.eos_token or ("<|im_end|>" if family == "qwen" else "")

    # Скоринг требует ПРАВОГО паддинга: префикс (промпт+картинка) выровнен слева,
    # значит позиции ≥ plen — токены кандидата. Qwen по умолчанию паддит слева.
    _pad_saved = getattr(tokenizer, "padding_side", None)
    for _tok in (tokenizer, getattr(processor, "tokenizer", None)):
        if _tok is not None:
            _tok.padding_side = "right"

    n_total = len(test_dataset)
    max_eval = min(cfg.eval_max_samples, n_total)
    random.seed(42)
    eval_indices = sorted(random.sample(range(n_total), max_eval)) if n_total > max_eval else list(range(n_total))

    logger.info("EVALUATION (rank): %d/%d samples, image_mode=%s, cand_batch=%d",
                max_eval, n_total, image_mode, cand_batch)

    # shuffle: картинка от СЛЕДУЮЩЕГО запроса в eval-наборе (валидна, но не та).
    shuffle_lookup: Dict[int, str] = {}
    if image_mode == "shuffle":
        paths = [test_dataset[i][cfg.image_column] for i in eval_indices]
        shuffle_lookup = dict(zip(eval_indices, paths[1:] + paths[:1]))

    # Считаем логиты ТОЛЬКО для окна ответа (последние `keep` позиций), а не для
    # всей последовательности: full-vocab logits [B, ~1000, 152k] в fp32 = десятки
    # ГБ и OOM. logits_to_keep=k → lm_head только по последним k позициям.
    _keep_kw = {"name": None}   # какой kwarg поддерживает эта версия transformers

    def _forward_keep(enc, keep):
        if _keep_kw["name"] is not None:
            return model(**enc, **{_keep_kw["name"]: keep}).logits
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                out = model(**enc, **{kw: keep}).logits
                _keep_kw["name"] = kw
                return out
            except TypeError:
                continue
        _keep_kw["name"] = "__none__"          # старый API — режем полные логиты
        return model(**enc).logits[:, -keep:, :]

    @torch.no_grad()
    def _score(prompt_text, image, candidates):
        p = processor(text=[prompt_text], images=[image], return_tensors="pt")
        plen = int(p["input_ids"].shape[1])
        out = _np.full(len(candidates), -1e30, dtype=_np.float64)
        for s in range(0, len(candidates), cand_batch):
            chunk = candidates[s:s + cand_batch]
            texts = [f"{prompt_text}{c}{eos}" for c in chunk]
            enc = processor(text=texts, images=[image] * len(chunk),
                            return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            ids, attn = enc["input_ids"], enc["attention_mask"]
            L = int(ids.shape[1])
            keep = max(2, L - plen + 1)               # окно: позиции [plen-1 .. L-1]
            logits = _forward_keep(enc, keep).float()  # [B, keep, V] — маленький
            # logits[:, :-1] на позициях [plen-1 .. L-2] предсказывают ids[:, plen:]
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, plen:]
            m = attn[:, plen:] > 0
            Lt = min(lp.shape[1], tgt.shape[1])
            lp, tgt, m = lp[:, :Lt], tgt[:, :Lt], m[:, :Lt]
            tok_lp = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            norm = ((tok_lp * m).sum(-1) / m.sum(-1).clamp_min(1)).float().cpu().numpy()
            for j, v in enumerate(norm):
                out[s + j] = float(v)
            del logits, lp
        return out

    ranks: List[float] = []
    cand_sizes: List[int] = []
    detailed: List[Dict[str, Any]] = []
    t0 = time.time()
    for step, data_i in enumerate(eval_indices):
        sample = test_dataset[data_i]
        prov = candidate_provider(sample)
        if prov is None:
            continue
        candidates, filter_others, gold = prov
        if gold not in candidates or len(candidates) < 2:
            continue
        image = _load_eval_image(sample, cfg, image_mode, override_path=shuffle_lookup.get(data_i))
        prompt_text = format_eval_text(processor, family, sample[cfg.prompt_column])
        sc = _score(prompt_text, image, candidates)
        gpos = candidates.index(gold)
        gscore = sc[gpos]
        higher = sum(1 for i, c in enumerate(candidates)
                     if i != gpos and c not in filter_others and sc[i] > gscore)
        if realistic_ties:
            ties = sum(1 for i, c in enumerate(candidates)
                       if i != gpos and c not in filter_others and sc[i] == gscore)
            rank = 1 + higher + ties / 2.0
        else:
            rank = 1 + higher
        ranks.append(rank)
        cand_sizes.append(len(candidates))
        detailed.append({"idx": data_i, "gold": gold, "rank": rank, "n_cand": len(candidates)})
        if (step + 1) % 50 == 0 or step + 1 == len(eval_indices):
            sp = (step + 1) / max(time.time() - t0, 1e-6)
            logger.info("[%5d/%d] %.1f smp/s | scored=%d", step + 1, len(eval_indices), sp, len(ranks))

    try:
        cost = measure_token_cost(processor, tokenizer, test_dataset, cfg,
                                  cfg.prompt_column, image_mode)
    except Exception as exc:
        logger.warning("token cost measure failed: %s", exc)
        cost = {}

    r = np.array(ranks, dtype=float)
    rr = np.where(r > 0, 1.0 / r, 0.0)
    metrics: Dict[str, Any] = {
        "MRR": float(rr.mean()) if len(r) else 0.0,
        "Hits@1": float((r <= 1).mean()) * 100 if len(r) else 0.0,
        "Hits@3": float((r <= 3).mean()) * 100 if len(r) else 0.0,
        "Hits@10": float((r <= 10).mean()) * 100 if len(r) else 0.0,
        **cost,
        "n_evaluated": int(len(r)),
        "n_total_test": n_total,
        "mean_candidates": float(np.mean(cand_sizes)) if cand_sizes else 0.0,
        "eval_mode": "rank",
        "tie_break": "realistic" if realistic_ties else "optimistic",
        "image_mode": image_mode,
    }
    logger.info("=" * 65)
    logger.info("RESULTS rank (%d queries, %.1f min) | filtered", len(r), (time.time() - t0) / 60)
    logger.info("  MRR %.4f | H@1 %.1f | H@3 %.1f | H@10 %.1f | med.cand %.0f",
                metrics["MRR"], metrics["Hits@1"], metrics["Hits@3"], metrics["Hits@10"],
                float(np.median(cand_sizes)) if cand_sizes else 0.0)
    logger.info("=" * 65)

    suffix = "_rank" if image_mode == "real" else f"_rank_{image_mode}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    for fname, data in [(f"test_metrics{suffix}.json", metrics), (f"test_details{suffix}.json", detailed)]:
        with open(os.path.join(cfg.output_dir, fname), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    if _pad_saved is not None:
        for _tok in (tokenizer, getattr(processor, "tokenizer", None)):
            if _tok is not None:
                _tok.padding_side = _pad_saved
    return metrics


def evaluate_link_prediction_sample(
    model,
    processor,
    tokenizer,
    test_dataset,
    cfg: TrainConfig,
    valid_names: set,
    n_entities: int,
    filter_provider,
    image_mode: str = "real",
    prompt_key: Optional[str] = None,
    num_samples: int = 500,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    gen_batch: int = 50,
    max_new_tokens: int = 16,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """KGT5-протокол: ОТКРЫТОЕ сэмплирование ответов, ранг золотого среди
    сэмплированных сущностей (а не среди ~50 заданных кандидатов, как в rank).

    Для каждого запроса (h, r, ?):
      1. do_sample-генерация ``num_samples`` последовательностей (картинка по
         ``image_mode``, промпт из колонки ``prompt_key``);
      2. декодируем, нормализуем, оставляем строки, совпавшие с ИМЕНЕМ реальной
         сущности (``valid_names``) — прочие (галлюцинации) отбрасываются;
      3. score сущности = лучшая (max) сумма лог-вероятностей её сэмплов;
      4. **filtered**: прочие истинные хвосты (h,r,·) убираются из ранжирования;
      5. ранг золотого = 1 + #{сущности со score > gold} (+ ½·равные — mean-rank);
         если золотой ни разу не сэмплирован → ранг = ``n_entities`` (= |E|).

    Это ровно то, чем меряют KGT5 / KGT5-context (сравнимо с их таблицами), в
    отличие от rank-режима (candidate-restricted, «лёгкий», не сравним с SOTA).

    ``filter_provider(sample) -> (gold_name, filter_others_set)`` или None.
    ``prompt_key`` выбирает колонку промпта (полный контекст vs минимальный) —
    так одним прогоном строится «зона B» (text-only / text-context / image-*).
    """
    prompt_key = prompt_key or cfg.prompt_column
    family = detect_family(cfg.model_name, cfg.model_family)
    device = next(model.parameters()).device
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    # Генерация decoder-only требует ЛЕВОГО паддинга (иначе позиции сдвигаются).
    _pad_saved = getattr(tokenizer, "padding_side", None)
    for _tok in (tokenizer, getattr(processor, "tokenizer", None)):
        if _tok is not None:
            _tok.padding_side = "left"

    n_total = len(test_dataset)
    max_eval = min(cfg.eval_max_samples, n_total)
    random.seed(42)   # тот же набор индексов, что в rank/generative → парные сравнения
    eval_indices = sorted(random.sample(range(n_total), max_eval)) if n_total > max_eval else list(range(n_total))

    tag = label or f"sample_{image_mode}_{prompt_key}"
    logger.info("EVALUATION (sample/KGT5): %d/%d | mode=%s prompt=%s | N=%d T=%.2f top_p=%.2f gb=%d [%s]",
                max_eval, n_total, image_mode, prompt_key, num_samples, temperature, top_p, gen_batch, tag)

    shuffle_lookup: Dict[int, str] = {}
    if image_mode == "shuffle":
        paths = [test_dataset[i][cfg.image_column] for i in eval_indices]
        shuffle_lookup = dict(zip(eval_indices, paths[1:] + paths[:1]))

    valid_norm = {normalize_answer(x) for x in valid_names}

    @torch.no_grad()
    def _sample_query(prompt_text, image):
        """→ dict{norm_entity_name: best_seq_logprob} по num_samples сэмплам."""
        ent_lp: Dict[str, float] = {}
        n_invalid = 0
        drawn = 0
        while drawn < num_samples:
            b = min(gen_batch, num_samples - drawn)
            enc = processor(text=[prompt_text] * b, images=[image] * b,
                            return_tensors="pt", padding=True, truncation=True,
                            max_length=cfg.max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            in_len = int(enc["input_ids"].shape[1])
            out = model.generate(
                **enc, do_sample=True, num_return_sequences=1,
                temperature=temperature, top_p=top_p,
                top_k=(top_k if top_k > 0 else None),
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
                renormalize_logits=True, pad_token_id=pad_id,
            )
            gen_ids = out.sequences[:, in_len:]                       # [b, gen]
            scores = torch.stack(out.scores, dim=1)                   # [b, gen, V] (log-softmax)
            lp = scores.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)  # [b, gen]
            # маска: токены до и включая первый eos (после — паддинг, не считаем)
            is_eos = gen_ids == eos_id
            prev_eos = torch.cumsum(is_eos.int(), dim=1) - is_eos.int()
            m = (prev_eos == 0)
            seq_lp = (lp * m).sum(-1).float().cpu().numpy()
            for row in range(gen_ids.shape[0]):
                txt = tokenizer.decode(gen_ids[row], skip_special_tokens=True).strip()
                norm = normalize_answer(txt)
                if not norm or norm not in valid_norm:
                    n_invalid += 1
                    continue
                v = float(seq_lp[row])
                if norm not in ent_lp or v > ent_lp[norm]:
                    ent_lp[norm] = v
            drawn += b
            del out, scores, lp
        return ent_lp, n_invalid

    ranks: List[float] = []
    distinct_counts: List[int] = []
    invalid_fracs: List[float] = []
    hit_sampled = 0
    detailed: List[Dict[str, Any]] = []
    t0 = time.time()

    for step, data_i in enumerate(eval_indices):
        sample = test_dataset[data_i]
        prov = filter_provider(sample)
        if prov is None:
            continue
        gold_name, filter_others = prov
        gold_norm = normalize_answer(gold_name)
        filt_norm = {normalize_answer(x) for x in filter_others} - {gold_norm}

        image = _load_eval_image(sample, cfg, image_mode, override_path=shuffle_lookup.get(data_i))
        prompt_text = format_eval_text(processor, family, sample[prompt_key])
        ent_lp, n_invalid = _sample_query(prompt_text, image)

        # filtered: убираем прочие истинные хвосты (кроме золотого)
        scored = {e: s for e, s in ent_lp.items() if e not in filt_norm}
        distinct_counts.append(len(scored))
        invalid_fracs.append(n_invalid / max(num_samples, 1))

        if gold_norm in scored:
            hit_sampled += 1
            gs = scored[gold_norm]
            greater = sum(1 for e, s in scored.items() if e != gold_norm and s > gs)
            equal = sum(1 for e, s in scored.items() if e != gold_norm and s == gs)
            rank = 1.0 + greater + equal / 2.0
        else:
            rank = float(n_entities)   # не сэмплирован → худший ранг = |E|
        ranks.append(rank)
        detailed.append({"idx": data_i, "gold": gold_name, "rank": rank,
                         "sampled": gold_norm in scored, "n_distinct": len(scored)})

        if (step + 1) % 20 == 0 or step + 1 == len(eval_indices):
            sp = (step + 1) / max(time.time() - t0, 1e-6)
            eta = (len(eval_indices) - step - 1) / max(sp, 1e-6) / 60
            logger.info("[%5d/%d] %.2f smp/s | sampled_gold=%d/%d | ETA %.0fm",
                        step + 1, len(eval_indices), sp, hit_sampled, len(ranks), eta)
        torch.cuda.empty_cache()

    # Стоимость подачи в токенах (ось «денег» кривой масштабирования).
    try:
        cost = measure_token_cost(processor, tokenizer, test_dataset, cfg,
                                  prompt_key, image_mode)
    except Exception as exc:
        logger.warning("token cost measure failed: %s", exc)
        cost = {}

    r = np.array(ranks, dtype=float)
    rr = np.where(r > 0, 1.0 / r, 0.0)
    metrics: Dict[str, Any] = {
        "MRR": float(rr.mean()) if len(r) else 0.0,
        "Hits@1": float((r <= 1).mean()) * 100 if len(r) else 0.0,
        "Hits@3": float((r <= 3).mean()) * 100 if len(r) else 0.0,
        "Hits@10": float((r <= 10).mean()) * 100 if len(r) else 0.0,
        **cost,
        "n_evaluated": int(len(r)),
        "n_total_test": n_total,
        "n_entities": int(n_entities),
        "sampled_gold_rate": float(hit_sampled / len(r)) * 100 if len(r) else 0.0,
        "mean_distinct_entities": float(np.mean(distinct_counts)) if distinct_counts else 0.0,
        "mean_invalid_frac": float(np.mean(invalid_fracs)) if invalid_fracs else 0.0,
        "eval_mode": "sample",
        "image_mode": image_mode,
        "prompt_key": prompt_key,
        "num_samples": num_samples,
        "temperature": temperature,
        "label": tag,
    }
    logger.info("=" * 65)
    logger.info("RESULTS sample/KGT5 [%s] (%d queries, %.1f min) | filtered, |E|=%d",
                tag, len(r), (time.time() - t0) / 60, n_entities)
    logger.info("  MRR %.4f | H@1 %.1f | H@3 %.1f | H@10 %.1f | gold-sampled %.1f%% | distinct %.0f | invalid %.0f%%",
                metrics["MRR"], metrics["Hits@1"], metrics["Hits@3"], metrics["Hits@10"],
                metrics["sampled_gold_rate"], metrics["mean_distinct_entities"],
                metrics["mean_invalid_frac"] * 100)
    logger.info("=" * 65)

    os.makedirs(cfg.output_dir, exist_ok=True)
    for fname, data in [(f"test_metrics_sample_{tag}.json", metrics),
                        (f"test_details_sample_{tag}.json", detailed)]:
        with open(os.path.join(cfg.output_dir, fname), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    if _pad_saved is not None:
        for _tok in (tokenizer, getattr(processor, "tokenizer", None)):
            if _tok is not None:
                _tok.padding_side = _pad_saved
    return metrics