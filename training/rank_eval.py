"""Filtered ranking для VLM — сопоставимо со стандартным протоколом KGC.

В отличие от генеративной оценки (`evaluate.py`, beam-search: ранг = позиция в
списке сгенерированных строк), здесь используется КЛАССИЧЕСКИЙ filtered-ranking,
как в KGE-литературе и в pykeen:

  1. для запроса (h, r, ?) кандидату-сущности c присваивается score(c) =
     длинно-нормированный log-lik имени c как ответа при данном изображении+промпте;
  2. кандидаты ранжируются по score;
  3. **filtered**: из ранжирования удаляются ДРУГИЕ истинные хвосты (h, r, ·)
     из train+valid+test, кроме золотого;
  4. ранг золотого = 1 + (#кандидатов со score выше золотого после фильтрации).

Метрики: MRR, Hits@1/3/10 + 95% ДИ — те же определения, что у KGE, поэтому
числа VLM и KGE (раздел T5) и числа из текста ВКР сопоставимы напрямую.

Два режима множества кандидатов:
  • "all"            — все сущности графа (точное соответствие стандарту FB15k-237,
                       дорого: ~14 265 кандидатов/запрос → запускать на выборке);
  • "relation_tails" — только хвосты, встречавшиеся у отношения r (type-constrained
                       filtered; медиана ~86 кандидатов → реально на полном тесте).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from graphvis_lp.config import TrainConfig
from graphvis_lp.training.families import build_train_texts, collate as family_collate

logger = logging.getLogger(__name__)


@torch.no_grad()
def score_candidates(
    model, processor, tokenizer, cfg: TrainConfig, family: str,
    image, prompt: str, candidates: Sequence[str], device,
    cand_batch: int = 64,
) -> np.ndarray:
    """Длинно-нормированный log-lik каждого кандидата-ответа (teacher-forcing).

    Изображение одно на все кандидаты; для простоты и корректности оно
    переэнкодируется в каждом чанке (вычислительно доминирует LM-forward).
    """
    scores = np.full(len(candidates), -1e30, dtype=np.float64)

    for start in range(0, len(candidates), cand_batch):
        chunk = candidates[start:start + cand_batch]
        feats: List[Dict[str, torch.Tensor]] = []
        ntok: List[int] = []

        for cand in chunk:
            _, full_text = build_train_texts(processor, tokenizer, family, prompt, cand)
            prompt_text, _ = build_train_texts(processor, tokenizer, family, prompt, "")
            full = processor(text=full_text, images=image, return_tensors="pt",
                             padding=False, truncation=True, max_length=cfg.max_length)
            penc = processor(text=prompt_text, images=image, return_tensors="pt",
                             padding=False, truncation=True, max_length=cfg.max_length)
            input_ids = full["input_ids"][0]
            plen = penc["input_ids"].shape[1]
            if plen >= input_ids.size(0):        # ответ обрезан
                feats.append(None); ntok.append(0); continue

            labels = input_ids.clone()
            labels[:plen] = -100
            feat = {"input_ids": input_ids,
                    "attention_mask": full["attention_mask"][0],
                    "labels": labels}
            if family == "qwen":
                feat["pixel_values"] = full["pixel_values"]
                feat["image_grid_thw"] = full["image_grid_thw"]
            else:
                feat["pixel_values"] = full["pixel_values"][0]
            feats.append(feat)
            ntok.append(int((labels != -100).sum().item()))

        valid = [(j, f) for j, f in enumerate(feats) if f is not None]
        if not valid:
            continue
        batch = family_collate(family, tokenizer, [f for _, f in valid], pad_to_multiple_of=None)
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits.float()
        labels = batch["labels"]
        # сдвиг: logits[:, :-1] предсказывают labels[:, 1:]
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = labels[:, 1:]
        mask = tgt != -100
        gathered = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        tot = (gathered * mask).sum(-1)                 # сумма log-prob ответа
        cnt = mask.sum(-1).clamp_min(1)
        norm = (tot / cnt).cpu().numpy()                # длинно-нормировка

        for (j, _), s in zip(valid, norm):
            scores[start + j] = float(s)

    return scores


def filtered_rank_eval(
    model, processor, tokenizer, cfg: TrainConfig, family: str,
    test_dataset,
    candidates_fn: Callable[[dict], List[str]],
    filter_fn: Callable[[dict], set],
    device,
    max_samples: Optional[int] = None,
    cand_batch: int = 64,
    seed: int = 42,
    log_every: int = 50,
    realistic_ties: bool = False,
) -> Dict[str, float]:
    """Возвращает filtered MRR / Hits@1,3,10 (+ 95% ДИ MRR)."""
    n_total = len(test_dataset)
    if max_samples and max_samples < n_total:
        idx = sorted(np.random.RandomState(seed).choice(n_total, size=max_samples, replace=False).tolist())
    else:
        idx = list(range(n_total))

    model.eval()
    from PIL import Image
    import os

    ranks: List[float] = []
    skipped = 0
    t0 = time.time()

    for c, i in enumerate(idx):
        row = test_dataset[int(i)]
        gold = row[cfg.answer_column].strip()
        cands = list(candidates_fn(row))
        if gold not in cands:
            cands.append(gold)
        if len(cands) < 2:
            skipped += 1
            continue

        raw = row[cfg.image_column].replace("\\", "/")
        path = os.path.join(cfg.image_root, raw) if cfg.image_root else raw
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (336, 336), (128, 128, 128))

        prompt = row[cfg.prompt_column]
        scores = score_candidates(model, processor, tokenizer, cfg, family,
                                  image, prompt, cands, device, cand_batch)

        gpos = cands.index(gold)
        gscore = scores[gpos]
        others = set(filter_fn(row)) - {gold}
        higher = sum(1 for k, cand in enumerate(cands)
                     if k != gpos and cand not in others and scores[k] > gscore)
        if realistic_ties:
            ties = sum(1 for k, cand in enumerate(cands)
                       if k != gpos and cand not in others and scores[k] == gscore)
            ranks.append(1 + higher + ties / 2.0)
        else:
            ranks.append(1 + higher)

        if (c + 1) % log_every == 0:
            el = time.time() - t0
            mrr = float(np.mean(1.0 / np.array(ranks)))
            logger.info("[%d/%d] MRR=%.4f | %.2f s/q | ~%d cand/q",
                        c + 1, len(idx), mrr, el / (c + 1), len(cands))

    r = np.array(ranks, dtype=float)
    rr = 1.0 / r
    ci = 1.96 * rr.std(ddof=1) / np.sqrt(len(rr)) if len(rr) > 1 else 0.0
    return {
        "MRR": float(rr.mean()),
        "MRR_ci95": float(ci),
        "Hits@1": float((r <= 1).mean()) * 100,
        "Hits@3": float((r <= 3).mean()) * 100,
        "Hits@10": float((r <= 10).mean()) * 100,
        "n_evaluated": int(len(r)),
        "n_skipped": int(skipped),
        "mean_candidates": float(np.mean([len(candidates_fn(test_dataset[int(i)])) for i in idx[:50]])),
    }
