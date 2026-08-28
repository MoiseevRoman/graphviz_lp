"""Обёртка GPU-этапа кривой масштабирования: бюджеты × условия автоматически.

Связывает три уже существующих кирпича в один прогон:

    rerender_dataset.py   → датасет на каждый бюджет k (число объектов)
    viability_gate.py     → обучение+eval (KGT5) на условие (visual/text/text_feat)
    scaling_curve.py      → стоимость в токенах + итоговая фигура

Раскладка результатов согласована с scaling_curve --mode plot:
    <ckpt_root>/k{K}/ctx_{cond}/test_metrics_sample_*.json

Условие → context_mode (viability_gate) один-в-один:
    visual     — картинка + minimal-промпт (наш метод)
    text       — соседи текстом + blank (≈ KGT5-context)
    text_feat  — соседи текстом + инженерные признаки движка + blank (гипотеза bias)
    both/both_feat/none — тоже поддержаны

Примеры:
    # полный прогон: собрать датасеты, обучить+оценить 3 условия, построить фигуру
    python scripts/run_scaling.py --budgets 1,2,4,8,16 --target_size 560 \
        --conditions visual,text,text_feat --train_n 4000 --eval_n 400 --num_samples 300

    # датасеты уже собраны — только оценка адаптеров + фигура
    python scripts/run_scaling.py --budgets 1,2,4,8,16 --skip_build --eval_only

    # посмотреть команды без запуска
    python scripts/run_scaling.py --budgets 1,2,4 --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY = sys.executable
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _p in (_ROOT, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run(cmd: List[str], dry: bool) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def _budgets(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


# Оверхед общей части промпта (Query + инструкция) — приблизительно, в токенах.
_QUERY_OVERHEAD = 25
# Оверхед формулировки визуального контекста ("The image shows …").
_VISUAL_CTX_OVERHEAD = 15


def cond_mean_tokens(ds_split: str, conditions: List[str], target_size: int,
                     model: str, min_px: int, max_px: int, n: int = 256) -> dict:
    """Среднее число ВХОДНЫХ токенов на пример по условиям (офлайн-прокси).

    Нужен для iso-token: чтобы уравнять train_n×tokens across conditions, надо
    знать токены каждой модальности ДО обучения. Текст считаем токенайзером
    (или len/4), картинку — точной формулой Qwen от target_size. Точность
    приблизительная — она лишь задаёт число примеров; viability потом логирует
    ТОЧНЫЙ бюджет обучения.
    """
    from datasets import load_from_disk
    from scaling_curve import text_tokens, _load_tokenizer, image_visual_tokens, _family

    ds = load_from_disk(ds_split)
    n = min(n, len(ds))
    tok = _load_tokenizer(model)
    fam = _family(model)
    img_tok = image_visual_tokens(fam, target_size, min_px, max_px)

    means = {}
    for cond in conditions:
        vals = []
        for i in range(n):
            row = ds[i]
            if cond in ("text", "text_feat"):
                col = "neighbors_feat" if cond == "text_feat" else "neighbors_text"
                ctx = row.get(col) or row.get("neighbors_text", "")
                vals.append(_QUERY_OVERHEAD + text_tokens(tok, ctx))
            elif cond in ("both", "both_feat"):
                col = "neighbors_feat" if cond == "both_feat" else "neighbors_text"
                ctx = row.get(col) or row.get("neighbors_text", "")
                vals.append(_QUERY_OVERHEAD + _VISUAL_CTX_OVERHEAD + text_tokens(tok, ctx) + img_tok)
            elif cond == "visual":
                vals.append(_QUERY_OVERHEAD + _VISUAL_CTX_OVERHEAD + img_tok)
            else:  # none
                vals.append(_QUERY_OVERHEAD)
        means[cond] = sum(vals) / max(len(vals), 1)
    return means


def iso_counts(means: dict, base_train_n: int, base_eval_n: int,
               token_budget: int, max_train_n: int) -> dict:
    """Per-condition (train_n, eval_n) под ОБЩИЙ токен-бюджет.

    Бюджет B (train): при token_budget=0 якорим на самую тяжёлую модальность —
    она сохраняет base_train_n, лёгкие получают БОЛЬШЕ примеров, чтобы суммарные
    токены совпали. eval_n масштабируем тем же коэффициентом.
    """
    heaviest = max(means.values()) if means else 1.0
    B = token_budget if token_budget > 0 else base_train_n * heaviest
    Be = base_eval_n * heaviest
    out = {}
    for cond, mt in means.items():
        mt = max(mt, 1.0)
        tn = int(round(B / mt))
        en = int(round(Be / mt))
        if max_train_n > 0:
            tn = min(tn, max_train_n)
        out[cond] = {"train_n": max(tn, 1), "eval_n": max(en, 1), "mean_tokens": mt}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--conditions", default="visual,text,text_feat",
                    help="через запятую: visual|text|text_feat|both|both_feat|none")
    ap.add_argument("--target_size", type=int, default=560, help="сторона рендера (Qwen 448..751)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--method", default="hybrid", help="ранжирование соседей для rerender")
    ap.add_argument("--workers", type=int, default=4, help="воркеров рендера")
    ap.add_argument("--feat_layout", action="store_true",
                    help="включить layout-координаты движка в neighbors_feat (дороже)")
    # обучение/оценка (passthrough в viability_gate)
    ap.add_argument("--train_n", type=int, default=4000)
    ap.add_argument("--eval_n", type=int, default=400)
    ap.add_argument("--build_limit", type=int, default=0,
                    help="сколько примеров на сплит рендерить (0 = авто = max(train_n, 2*eval_n)). "
                         "Иначе rerender соберёт ВЕСЬ сплит (~270k train) — почти всегда лишнее.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_mode", choices=["sample", "rank"], default="sample")
    ap.add_argument("--num_samples", type=int, default=300)
    ap.add_argument("--gen_batch", type=int, default=50)
    ap.add_argument("--sample_temperature", type=float, default=1.0)
    ap.add_argument("--train_vision", action="store_true",
                    help="разморозить зрительную башню (нужно, чтобы картинка МОГЛА работать)")
    # iso-token: равный ВЫЧИСЛИТЕЛЬНЫЙ бюджет across conditions
    ap.add_argument("--iso_tokens", action="store_true",
                    help="уравнять токен-бюджет по условиям: per-condition train_n/eval_n "
                         "подбираются так, чтобы train_n×tokens совпадало (тексто-тяжёлым — "
                         "меньше примеров, лёгкой картинке — больше). Иначе train_n одинаков "
                         "по ЧИСЛУ примеров, но НЕ по токенам.")
    ap.add_argument("--token_budget", type=int, default=0,
                    help="целевой бюджет токенов/пример-эквивалент (0 = авто: якорь на самую "
                         "тяжёлую модальность, она сохраняет --train_n)")
    ap.add_argument("--qwen_min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--qwen_max_pixels", type=int, default=720 * 28 * 28)
    ap.add_argument("--max_train_n", type=int, default=0,
                    help="потолок train_n на условие в iso-режиме (0 = без потолка); "
                         "защита от гигантского train_n у дешёвой по токенам модальности")
    # пути
    ap.add_argument("--out_base", default=os.path.join(_ROOT, "graphvis_scaling"),
                    help="датасеты по бюджетам: <out_base>/k{K}/datasets")
    ap.add_argument("--ckpt_root", default=os.path.join(_ROOT, "graphvis_ckpts_scaling"),
                    help="чекпойнты/метрики: <ckpt_root>/k{K}/ctx_{cond}")
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "FB15k-237"))
    # управление этапами
    ap.add_argument("--skip_build", action="store_true", help="не перерендеривать датасеты")
    ap.add_argument("--eval_only", action="store_true", help="не обучать — грузить адаптер и только оценивать")
    ap.add_argument("--skip_plot", action="store_true", help="не строить cost/plot в конце")
    ap.add_argument("--dry_run", action="store_true", help="печатать команды, ничего не запуская")
    args = ap.parse_args()

    budgets = _budgets(args.budgets)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    dry = args.dry_run

    print(f"=== SCALING RUN: budgets={budgets} conditions={conditions} "
          f"target_size={args.target_size} model={args.model} ===")

    for k in budgets:
        k_root = os.path.join(args.out_base, f"k{k}")
        ds_root = os.path.join(k_root, "datasets")

        # ── 1. датасет бюджета ──
        if not args.skip_build:
            test_split = os.path.join(ds_root, "stage2_test")
            if os.path.isdir(test_split):
                print(f"[k={k}] датасет уже есть: {ds_root} (пропуск build)")
            else:
                print(f"[k={k}] build → {k_root}")
                build_limit = args.build_limit or max(args.train_n, 2 * args.eval_n)
                cmd = [_PY, os.path.join(_SCRIPTS, "rerender_dataset.py"),
                       "--method", args.method, "--k", str(k),
                       "--target_size", str(args.target_size),
                       # stage2_valid намеренно НЕ рендерится per-budget: scaling-eval
                       # идёт только на stage2_test, а filtered-ранжирование и вселенная
                       # сущностей берутся из сырых троек data_dir (train+valid+test) через
                       # build_name_maps/build_entity_universe — перерендеренный valid не нужен.
                       "--splits", "stage2_train,stage2_test",
                       "--limit", str(build_limit),
                       "--workers", str(args.workers), "--out_root", k_root]
                if args.feat_layout:
                    cmd.append("--feat_layout")
                _run(cmd, dry)

        # ── matched-info: доп. текстовые сериализации подграфа (те же рёбра) ──
        # text_adjlist/text_matrix требуют колонок neighbors_adjlist/neighbors_matrix,
        # которых rerender не создаёт. Добавляем идемпотентно (пропуск, если уже есть),
        # чтобы viability_gate не откатился молча к edge-list и сравнение не выродилось.
        _enc = [e for c, e in (("text_adjlist", "adjlist"), ("text_matrix", "matrix"))
                if c in conditions]
        if _enc and not args.skip_build:
            _run([_PY, os.path.join(_SCRIPTS, "add_text_encodings.py"),
                  "--budgets", str(k), "--data_base", args.out_base,
                  "--splits", "stage2_train,stage2_test",
                  "--encodings", ",".join(_enc)], dry)

        # ── iso-token: per-condition train_n/eval_n под общий бюджет ──
        iso = None
        if args.iso_tokens and not (args.dry_run and not os.path.isdir(os.path.join(ds_root, "stage2_train"))):
            try:
                means = cond_mean_tokens(
                    os.path.join(ds_root, "stage2_train"), conditions,
                    args.target_size, args.model, args.qwen_min_pixels, args.qwen_max_pixels)
                iso = iso_counts(means, args.train_n, args.eval_n,
                                 args.token_budget, args.max_train_n)
                print(f"[k={k}] iso-token бюджет (равные токены на тест):")
                for c, v in iso.items():
                    print(f"        {c:<11} tokens/ex≈{v['mean_tokens']:6.0f} → "
                          f"train_n={v['train_n']}  eval_n={v['eval_n']}")
                if not dry:
                    iso_dir = os.path.join(args.ckpt_root, f"k{k}")
                    os.makedirs(iso_dir, exist_ok=True)
                    import json as _json
                    _json.dump({"means": means, "counts": iso,
                                "token_budget": args.token_budget},
                               open(os.path.join(iso_dir, "iso_budget.json"), "w"),
                               indent=2, ensure_ascii=False)
            except Exception as exc:
                print(f"[k={k}] iso-token не удалось посчитать ({exc}); откат к фиксированным train_n/eval_n")
                iso = None

        # ── 2. условия ──
        for cond in conditions:
            outdir = os.path.join(args.ckpt_root, f"k{k}", f"ctx_{cond}")
            os.makedirs(outdir, exist_ok=True) if not dry else None
            tn = iso[cond]["train_n"] if iso and cond in iso else args.train_n
            en = iso[cond]["eval_n"] if iso and cond in iso else args.eval_n
            print(f"[k={k}] условие «{cond}» (train_n={tn}, eval_n={en}) → {outdir}")
            cmd = [_PY, os.path.join(_SCRIPTS, "viability_gate.py"),
                   "--model", args.model,
                   "--dataset_root", ds_root,
                   "--data_dir", args.data_dir,
                   "--output_dir", outdir,
                   "--out", os.path.join(outdir, "report.json"),
                   "--eval_mode", args.eval_mode,
                   "--context_mode", cond,
                   "--train_n", str(tn),
                   "--eval_n", str(en),
                   "--epochs", str(args.epochs),
                   "--num_samples", str(args.num_samples),
                   "--gen_batch", str(args.gen_batch),
                   "--sample_temperature", str(args.sample_temperature)]
            if args.train_vision:
                cmd.append("--train_vision")
            if args.eval_only:
                cmd.append("--eval_only")
            _run(cmd, dry)

    # ── 3. стоимость токенов + фигура ──
    if not args.skip_plot:
        cost_out = os.path.join(args.out_base, "cost.json")
        fig_out = os.path.join(args.out_base, "scaling_curve.png")
        print("[curve] cost + plot")
        _run([_PY, os.path.join(_SCRIPTS, "scaling_curve.py"),
              "--mode", "cost", "--budgets", args.budgets,
              "--target_size", str(args.target_size), "--model_name", args.model,
              "--out_base", args.out_base, "--text_field", "neighbors_text",
              "--cost_out", cost_out], dry)
        _run([_PY, os.path.join(_SCRIPTS, "scaling_curve.py"),
              "--mode", "plot", "--budgets", args.budgets,
              "--conditions", args.conditions, "--metric", "MRR",
              "--eval_mode", args.eval_mode,
              "--ckpt_root", args.ckpt_root, "--cost_out", cost_out,
              "--fig_out", fig_out], dry)

    print("=== SCALING RUN готово ===")
    if not args.skip_plot and not dry:
        print(f"Фигура: {os.path.join(args.out_base, 'scaling_curve.png')}")


if __name__ == "__main__":
    main()
