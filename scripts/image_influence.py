"""Насколько информация с изображения РЕАЛЬНО доходит до языковой модели.

Три независимых пробы (работают и для LLaVA, и для Qwen2.5-VL):

  ПРОБА B — причинная чувствительность выхода (teacher-forcing, без beam):
      для каждого запроса считаем NLL золотого ответа и распределение
      следующего токена при РЕАЛЬНОМ изображении и при абляциях
      (blank / noise / shuffle). Метрики:
        ΔNLL = NLL(ablated) − NLL(real)   > 0  ⇒ картинка помогает;
        симметричная KL между предсказаниями real и ablated
                                          > 0  ⇒ картинка меняет выход LM.
      Не зависит от beam-search и покрытия ответов — чистый сигнал
      «информация дошла до LM».

  ПРОБА C — доля внимания на визуальные токены:
      прогон с output_attentions (attn_implementation="eager"); на позициях,
      порождающих ответ, считаем долю attention-массы, уходящую на
      визуальные токены, по слоям. Сравниваем с «случайной» долей
      (= доля визуальных токенов в контексте). attn_image ≫ chance ⇒
      модель активно читает изображение.

  ПРОБА A — поведенческая абляция MRR (опционально, --behavioral):
      полный link-prediction в режимах real/blank/noise/shuffle
      (переиспользует evaluate_link_prediction).

Usage:
    python -m graphvis_lp.scripts.image_influence [CHECKPOINT] \
        [--n 100] [--split stage2_test] [--behavioral]

CHECKPOINT — путь к обученному LoRA-адаптеру (можно опустить: тогда
оценивается базовая модель, сигнал слабее, но пробы валидны).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import sys as _sys
import types as _types

# Windows cp1251-консоль роняла процесс на печати 'Δ' (ΔNLL) ПОСЛЕ подсчёта проб,
# но ДО json.dump → результаты терялись. Переводим вывод на utf-8.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Пакет импортируется как `graphvis_lp`, но папка может называться иначе
# (напр. graphviz_lp). Кладём корень репо на путь и вешаем алиас пакета —
# тогда graphvis_lp.* резолвится независимо от имени папки.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
if "graphvis_lp" not in _sys.modules:
    _pkg = _types.ModuleType("graphvis_lp"); _pkg.__path__ = [_ROOT]
    _sys.modules["graphvis_lp"] = _pkg

from graphvis_lp.config import TrainConfig
from graphvis_lp.training.families import (
    build_train_texts,
    collate as family_collate,
    detect_family,
    image_position_mask,
    image_token_id,
)

logger = logging.getLogger(__name__)

MODES = ["blank", "noise", "shuffle"]


# ──────────────────────────────────────────────────────────────────
# Изображения
# ──────────────────────────────────────────────────────────────────
def _resolve_path(cfg: TrainConfig, raw: str) -> str:
    rel = raw.replace("\\", "/")
    return os.path.join(cfg.image_root, rel) if cfg.image_root else rel


def _load_real(cfg: TrainConfig, sample) -> Image.Image:
    try:
        return Image.open(_resolve_path(cfg, sample[cfg.image_column])).convert("RGB")
    except Exception:
        return Image.new("RGB", (336, 336), (128, 128, 128))


def _ablate(mode: str, real: Image.Image, other: Optional[Image.Image]) -> Image.Image:
    """Абляционное изображение ТОГО ЖЕ размера, что и настоящее.

    Одинаковый размер критичен для Qwen: гарантирует идентичную сетку
    визуальных токенов ⇒ input_ids совпадают ⇒ позиции сопоставимы.
    """
    size = real.size
    if mode == "blank":
        return Image.new("RGB", size, (255, 255, 255))
    if mode == "noise":
        arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")
    if mode == "shuffle":
        base = other if other is not None else real
        return base.convert("RGB").resize(size)
    raise ValueError(mode)


# ──────────────────────────────────────────────────────────────────
# Построение батча из одного примера (переиспользует путь обучения)
# ──────────────────────────────────────────────────────────────────
def _build_batch(cfg, processor, tokenizer, family, prompt, answer, image, device):
    """Возвращает (batch_on_device, answer_query_idx, answer_target_idx) или None."""
    prompt_text, full_text = build_train_texts(processor, tokenizer, family, prompt, answer)

    full = processor(text=full_text, images=image, return_tensors="pt",
                     padding=False, truncation=True, max_length=cfg.max_length)
    prompt_enc = processor(text=prompt_text, images=image, return_tensors="pt",
                           padding=False, truncation=True, max_length=cfg.max_length)

    input_ids = full["input_ids"][0]
    attention_mask = full["attention_mask"][0]
    prompt_len = prompt_enc["input_ids"].shape[1]
    if prompt_len >= input_ids.size(0):
        return None  # ответ обрезан truncation

    labels = input_ids.clone()
    labels[:prompt_len] = -100

    feat: Dict[str, Any] = {
        "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
    }
    if family == "qwen":
        feat["pixel_values"] = full["pixel_values"]
        feat["image_grid_thw"] = full["image_grid_thw"]
    else:
        feat["pixel_values"] = full["pixel_values"][0]

    batch = family_collate(family, tokenizer, [feat], pad_to_multiple_of=None)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Позиции ответа: labels != -100. logits[i] предсказывает input_ids[i+1],
    # значит query-позиция для ответного токена t = t-1.
    ans_target = torch.nonzero(batch["labels"][0] != -100, as_tuple=False).flatten()
    ans_query = ans_target - 1
    return batch, ans_query, ans_target


# ──────────────────────────────────────────────────────────────────
# ПРОБА B: причинная чувствительность
# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def _forward_logits(model, batch):
    fwd = {k: v for k, v in batch.items() if k != "labels"}
    return model(**fwd).logits[0].float()  # (seq, vocab)


def probe_causal(model, processor, tokenizer, cfg, ds, indices, family, device):
    logger.info("ПРОБА B: причинная чувствительность (%d примеров)…", len(indices))
    agg = {m: {"dnll": [], "kl": []} for m in MODES}
    n_used = 0

    for k, i in enumerate(indices):
        s = ds[int(i)]
        prompt, answer = s[cfg.prompt_column], s[cfg.answer_column]
        real = _load_real(cfg, s)
        other = _load_real(cfg, ds[int(indices[(k + 1) % len(indices)])])

        built = _build_batch(cfg, processor, tokenizer, family, prompt, answer, real, device)
        if built is None:
            continue
        batch_r, q_idx, t_idx = built
        logits_r = _forward_logits(model, batch_r)
        logp_r = F.log_softmax(logits_r[q_idx], dim=-1)
        gold = batch_r["input_ids"][0][t_idx]
        nll_r = -logp_r.gather(-1, gold.unsqueeze(-1)).squeeze(-1).mean().item()

        for mode in MODES:
            img = _ablate(mode, real, other)
            built_a = _build_batch(cfg, processor, tokenizer, family, prompt, answer, img, device)
            if built_a is None:
                continue
            batch_a, qa, ta = built_a
            # Требуем совпадения позиций (иначе KL несопоставим).
            if qa.shape != q_idx.shape or not torch.equal(qa, q_idx):
                continue
            logits_a = _forward_logits(model, batch_a)
            logp_a = F.log_softmax(logits_a[q_idx], dim=-1)
            nll_a = -logp_a.gather(-1, gold.unsqueeze(-1)).squeeze(-1).mean().item()

            p_r, p_a = logp_r.exp(), logp_a.exp()
            kl = 0.5 * (
                (p_r * (logp_r - logp_a)).sum(-1) + (p_a * (logp_a - logp_r)).sum(-1)
            ).mean().item()

            agg[mode]["dnll"].append(nll_a - nll_r)
            agg[mode]["kl"].append(kl)

        n_used += 1
        if (k + 1) % 25 == 0:
            logger.info("  …%d/%d", k + 1, len(indices))

    out = {"n_used": n_used}
    for m in MODES:
        d, kl = agg[m]["dnll"], agg[m]["kl"]
        out[m] = {
            "dNLL_mean": float(np.mean(d)) if d else 0.0,
            "dNLL_sem": float(np.std(d) / np.sqrt(len(d))) if d else 0.0,
            "symKL_mean": float(np.mean(kl)) if kl else 0.0,
            "frac_helped": float(np.mean(np.array(d) > 0)) if d else 0.0,
            "n": len(d),
        }
    return out


# ──────────────────────────────────────────────────────────────────
# ПРОБА C: доля внимания на визуальные токены
# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_attention(model, processor, tokenizer, cfg, ds, indices, family, device):
    logger.info("ПРОБА C: доля внимания на изображение (%d примеров)…", len(indices))
    img_id = image_token_id(model, tokenizer, family)

    per_layer: List[List[float]] = []
    chance: List[float] = []
    n_used = 0

    for k, i in enumerate(indices):
        s = ds[int(i)]
        prompt, answer = s[cfg.prompt_column], s[cfg.answer_column]
        real = _load_real(cfg, s)
        built = _build_batch(cfg, processor, tokenizer, family, prompt, answer, real, device)
        if built is None:
            continue
        batch, q_idx, _ = built
        if q_idx.numel() == 0:
            continue

        out = model(**{k2: v for k2, v in batch.items() if k2 != "labels"},
                    output_attentions=True)
        attns = out.attentions  # tuple(L) × (1, H, seq, seq)
        if attns is None or attns[0] is None:
            logger.warning("attentions=None — нужен attn_implementation='eager'. Пропуск пробы C.")
            return {"error": "no_attentions"}

        img_mask = image_position_mask(batch["input_ids"][0], img_id)  # (seq,)
        img_cols = img_mask.nonzero(as_tuple=False).flatten()
        if img_cols.numel() == 0:
            continue
        seq = batch["input_ids"].shape[1]
        chance.append(float(img_cols.numel()) / seq)

        layer_fracs = []
        for layer_attn in attns:
            a = layer_attn[0]                      # (H, seq, seq)
            a_q = a[:, q_idx, :]                    # (H, |ans|, seq)
            img_mass = a_q[:, :, img_cols].sum(-1)  # (H, |ans|)
            total = a_q.sum(-1).clamp_min(1e-9)     # (H, |ans|)
            layer_fracs.append((img_mass / total).mean().item())
        per_layer.append(layer_fracs)
        n_used += 1
        if (k + 1) % 25 == 0:
            logger.info("  …%d/%d", k + 1, len(indices))

    if not per_layer:
        return {"error": "no_samples"}

    arr = np.array(per_layer)               # (N, L)
    layer_mean = arr.mean(0).tolist()
    return {
        "n_used": n_used,
        "n_layers": arr.shape[1],
        "attn_image_frac_mean": float(arr.mean()),
        "attn_image_frac_per_layer": layer_mean,
        "chance_frac_mean": float(np.mean(chance)),
        "ratio_over_chance": float(arr.mean() / max(np.mean(chance), 1e-9)),
    }


# ──────────────────────────────────────────────────────────────────
# ПРОБА A: поведенческая абляция MRR (опционально)
# ──────────────────────────────────────────────────────────────────
def probe_behavioral(model, processor, tokenizer, cfg, test):
    from graphvis_lp.training.evaluate import evaluate_link_prediction
    res = {}
    for mode in ["real", "blank", "noise", "shuffle"]:
        res[mode] = evaluate_link_prediction(
            model, processor, tokenizer, test, cfg, image_mode=mode,
        )
    return res


# ──────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=None, help="путь к LoRA-адаптеру")
    ap.add_argument("--n", type=int, default=100, help="число примеров для проб B/C")
    ap.add_argument("--split", default="stage2_test")
    ap.add_argument("--dataset_root", default=None,
                    help="датасет чекпойнта (напр. hybrid8_672\\datasets); по умолчанию — из TrainConfig")
    ap.add_argument("--out", default=None,
                    help="куда писать json (по умолчанию <output_dir>/image_influence.json)")
    ap.add_argument("--model", default=None,
                    help="базовая модель; по умолчанию читается из adapter_config.json чекпойнта")
    ap.add_argument("--behavioral", action="store_true", help="запустить также пробу A (MRR)")
    args = ap.parse_args()

    from datasets import load_from_disk
    from peft import PeftModel

    from graphvis_lp.training.model import build_model_and_processor

    cfg = TrainConfig()
    cfg.use_lora = False
    cfg.attn_implementation = "eager"  # обязательно для пробы C
    if args.dataset_root:
        cfg.dataset_root = args.dataset_root
    # Датасеты hybrid8 хранят ОТНОСИТЕЛЬНЫЙ image_path (от корня репо) → резолвим от _ROOT.
    cfg.image_root = _ROOT
    # База ОБЯЗАНА совпасть с той, на которой обучен адаптер, иначе size mismatch
    # (дефолт TrainConfig — LLaVA-7B, а наши адаптеры — Qwen2.5-VL-3B). Берём из адаптера.
    if args.model:
        cfg.model_name = args.model
    elif args.checkpoint:
        _acfg = os.path.join(args.checkpoint, "adapter_config.json")
        if os.path.exists(_acfg):
            _base = json.load(open(_acfg, encoding="utf-8")).get("base_model_name_or_path")
            if _base:
                cfg.model_name = _base
                logger.info("Базовая модель из adapter_config.json: %s", _base)
    cfg.model_family = "auto"
    family = detect_family(cfg.model_name, cfg.model_family)

    model, processor, tokenizer = build_model_and_processor(cfg)
    if args.checkpoint:
        logger.info("Загрузка адаптера: %s", args.checkpoint)
        model = PeftModel.from_pretrained(model, args.checkpoint)
    else:
        logger.warning("Чекпоинт не указан — базовая модель (сигнал слабее).")
    model.eval()
    device = next(model.parameters()).device

    ds = load_from_disk(os.path.join(cfg.dataset_root, args.split))
    n = min(args.n, len(ds))
    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(len(ds), size=n, replace=False).tolist())

    report: Dict[str, Any] = {
        "model": cfg.model_name, "family": family,
        "checkpoint": args.checkpoint, "split": args.split, "n": n,
    }
    report["probe_B_causal"] = probe_causal(
        model, processor, tokenizer, cfg, ds, indices, family, device)
    report["probe_C_attention"] = probe_attention(
        model, processor, tokenizer, cfg, ds, indices, family, device)
    if args.behavioral:
        report["probe_A_behavioral"] = probe_behavioral(
            model, processor, tokenizer, cfg, ds)

    # ── Печать ──
    b = report["probe_B_causal"]
    print("\n" + "=" * 66)
    print("ПРОБА B — причинная чувствительность выхода LM (real vs абляция)")
    print(f"{'mode':<9} {'ΔNLL':>10} {'±sem':>8} {'symKL':>10} {'helped%':>9} {'n':>5}")
    print("-" * 66)
    for m in MODES:
        r = b[m]
        print(f"{m:<9} {r['dNLL_mean']:>+10.4f} {r['dNLL_sem']:>8.4f} "
              f"{r['symKL_mean']:>10.4f} {100*r['frac_helped']:>8.0f}% {r['n']:>5}")
    print("ΔNLL>0 и symKL>0 ⇒ картинка меняет выход LM (информация доходит).")

    c = report["probe_C_attention"]
    print("\n" + "=" * 66)
    print("ПРОБА C — доля внимания ответных позиций на визуальные токены")
    if "error" in c:
        print(f"  недоступно: {c['error']}")
    else:
        print(f"  средняя доля внимания на изображение : {c['attn_image_frac_mean']:.4f}")
        print(f"  случайная доля (=доля виз. токенов)  : {c['chance_frac_mean']:.4f}")
        print(f"  отношение к случайной                : ×{c['ratio_over_chance']:.2f}")
        pl = c["attn_image_frac_per_layer"]
        top = sorted(range(len(pl)), key=lambda j: pl[j], reverse=True)[:5]
        print(f"  топ-слои по вниманию к картинке      : "
              + ", ".join(f"L{j}={pl[j]:.3f}" for j in top))
        print("  ×>1 ⇒ модель смотрит на изображение чаще, чем случайно.")
    print("=" * 66)

    path = args.out or os.path.join(cfg.output_dir, "image_influence.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()
