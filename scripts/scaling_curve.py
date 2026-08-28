"""Кривая масштабирования позиции: качество vs число отрисованных объектов,
со второй осью — СТОИМОСТЬ подачи в токенах. Главная фигура исследования.

Идея (из обсуждения с научруком): построить зависимость качества LP от числа
объектов в контексте и наложить стоимость (токены/деньги). Ожидаемая форма:
плато → рост → плато/спад. Одновременно видно, где картинка перестаёт окупаться
по токенам. Кривую стоимости картинки имеет смысл строить ТОЛЬКО на Qwen
(dynamic resolution); у LLaVA визуальная стоимость фиксирована (576) — см.
token_cost.py.

Три независимо запускаемых режима:

  build : для каждого бюджета k перерендерить датасет (rerender_dataset.py --k)
          в целевом размере Qwen → out_base/k{K}/…  (нужны Neo4j + dot).
  cost  : офлайн — по собранным датасетам посчитать среднее число объектов и
          стоимость в токенах на модальность (картинка: qwen_visual_tokens от
          target_size; текст: токенайзер по text_field). Пишет cost.json.
  plot  : собрать метрики качества по бюджетам/условиям (test_metrics_sample_*)
          + cost.json → фигура quality-vs-objects с осью токенов и CSV.

  python scripts/scaling_curve.py --mode build --budgets 1,2,4,8,16 --target_size 560
  python scripts/scaling_curve.py --mode cost  --budgets 1,2,4,8,16 --target_size 560
  python scripts/scaling_curve.py --mode plot  --budgets 1,2,4,8,16 --conditions visual,text,text_feat
  python scripts/scaling_curve.py --selftest        # синтетика: проверка логики+фигуры
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from token_cost import qwen_visual_tokens, LLAVA_VISUAL_TOKENS  # noqa: E402


# ────────────────────────────────────────────────────────────────
# Общие помощники (чистые, тестируются на синтетике)
# ────────────────────────────────────────────────────────────────
def count_objects(text: str) -> int:
    """Число объектов (рёбер) в текстовом представлении окрестности.

    Работает и для neighbors_text (по строкам-фактам), и для prompt_triples
    (по строкам-триплетам вида ``(a, r, b)``)."""
    if not text:
        return 0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    tri = [ln for ln in lines if ln.strip().startswith("(") and "," in ln]
    return len(tri) if tri else len(lines)


def heuristic_tokens(text: str) -> int:
    """Грубая оценка числа токенов без токенайзера (~4 символа/токен)."""
    return max(1, round(len(text) / 4))


def text_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return heuristic_tokens(text)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def image_visual_tokens(
    family: str, side: int, min_pixels: int, max_pixels: int,
) -> int:
    """Визуальные токены картинки стороной ``side`` px для семейства модели."""
    if family == "qwen":
        return qwen_visual_tokens(side, side, min_pixels, max_pixels)
    return LLAVA_VISUAL_TOKENS


def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        print(f"[cost] токенайзер '{model_name}' недоступен ({exc}); эвристика len/4")
        return None


def _family(model_name: str) -> str:
    n = model_name.lower()
    return "qwen" if ("qwen" in n and "vl" in n) else "llava"


# ────────────────────────────────────────────────────────────────
# build
# ────────────────────────────────────────────────────────────────
def run_build(args) -> None:
    budgets = _parse_budgets(args.budgets)
    for k in budgets:
        out_root = os.path.join(args.out_base, f"k{k}")
        cmd = [
            sys.executable, os.path.join(_ROOT, "scripts", "rerender_dataset.py"),
            "--method", args.method, "--k", str(k),
            "--target_size", str(args.target_size),
            "--splits", args.splits, "--workers", str(args.workers),
            "--out_root", out_root,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print(f"[build] k={k} → {out_root}\n        {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


# ────────────────────────────────────────────────────────────────
# cost (офлайн)
# ────────────────────────────────────────────────────────────────
def run_cost(args) -> Dict[str, Dict]:
    from datasets import load_from_disk

    budgets = _parse_budgets(args.budgets)
    fam = _family(args.model_name)
    tok = _load_tokenizer(args.model_name)
    min_px, max_px = args.qwen_min_pixels, args.qwen_max_pixels
    img_tok = image_visual_tokens(fam, args.target_size, min_px, max_px)

    out: Dict[str, Dict] = {"meta": {
        "family": fam, "target_size": args.target_size,
        "image_visual_tokens": img_tok, "text_field": args.text_field,
    }, "budgets": {}}

    for k in budgets:
        path = os.path.join(args.out_base, f"k{k}", "datasets", args.split)
        if not os.path.exists(path):
            print(f"[cost] нет датасета: {path} — пропуск k={k}")
            continue
        ds = load_from_disk(path)
        n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
        objs, ttoks = [], []
        for i in range(n):
            row = ds[i]
            field = row.get(args.text_field) or ""
            objs.append(count_objects(field))
            ttoks.append(text_tokens(tok, field))
        mean_obj = sum(objs) / max(len(objs), 1)
        mean_txt = sum(ttoks) / max(len(ttoks), 1)
        out["budgets"][str(k)] = {
            "k": k,
            "mean_objects": mean_obj,
            "text_tokens": mean_txt,        # стоимость текстовой модальности
            "image_tokens": img_tok,        # стоимость картиночной (const по k у Qwen@side)
            "n": n,
        }
        print(f"[cost] k={k:>3}  objects≈{mean_obj:5.2f}  text_tok≈{mean_txt:6.1f}  img_tok={img_tok}")

    os.makedirs(os.path.dirname(args.cost_out) or ".", exist_ok=True)
    json.dump(out, open(args.cost_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[cost] сохранено: {args.cost_out}")
    return out


# ────────────────────────────────────────────────────────────────
# plot
# ────────────────────────────────────────────────────────────────
# Паттерны метрик по режиму оценки (порядок = приоритет в auto).
_METRIC_PATTERNS = {
    "sample": "test_metrics_sample_*.json",
    "rank": "test_metrics_rank*.json",
}


def _metric_path(ckpt_root: str, k: int, cond: str, eval_mode: str) -> Optional[str]:
    """Путь к файлу метрик для бюджета+условия при заданном eval_mode.

    eval_mode ∈ {rank, sample}: ищет ТОЛЬКО соответствующий паттерн. Если в
    папке >1 подходящего файла — падает (не молча sorted()[0])."""
    d = os.path.join(ckpt_root, f"k{k}", f"ctx_{cond}")
    hits = sorted(glob.glob(os.path.join(d, _METRIC_PATTERNS[eval_mode])))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            f"неоднозначные метрики режима '{eval_mode}' в {d}: {hits}; "
            f"оставьте один файл или задайте явный --eval_mode/тег")
    return hits[0]


def _resolve_eval_mode(ckpt_root: str, budgets: List[int], conds: List[str]) -> str:
    """auto: определить единый режим по всем ctx-папкам; при смешении — падать.

    Если файлы РАЗНЫХ режимов найдены (в разных папках или оба в одной) — метрики
    несравнимы (rank candidate-restricted vs sample/KGT5), поэтому явно падаем со
    списком конфликтующих путей, а не молча берём первый."""
    present: Dict[str, List[str]] = {m: [] for m in _METRIC_PATTERNS}
    for c in conds:
        for k in budgets:
            d = os.path.join(ckpt_root, f"k{k}", f"ctx_{c}")
            for m, pat in _METRIC_PATTERNS.items():
                present[m] += sorted(glob.glob(os.path.join(d, pat)))
    found = [m for m in _METRIC_PATTERNS if present[m]]
    if len(found) > 1:
        detail = "; ".join(f"{m}: {present[m]}" for m in found)
        raise RuntimeError(
            f"смешаны режимы оценки ({', '.join(found)}) — метрики несравнимы; "
            f"задайте --eval_mode явно. Найдено: {detail}")
    return found[0] if found else "rank"  # дефолт совместим: реальные папки — rank


def _find_metric(ckpt_root: str, k: int, cond: str, eval_mode: str) -> Optional[dict]:
    """Загружает метрики для бюджета+условия в режиме eval_mode (rank|sample)."""
    p = _metric_path(ckpt_root, k, cond, eval_mode)
    if p is None:
        return None
    return json.load(open(p, encoding="utf-8"))


def run_plot(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = _parse_budgets(args.budgets)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    metric = args.metric

    # Единый режим метрик: явный --eval_mode или авто-детект (падает при смешении).
    eval_mode = args.eval_mode if args.eval_mode != "auto" else _resolve_eval_mode(
        args.ckpt_root, budgets, conds)

    cost = json.load(open(args.cost_out, encoding="utf-8")) if os.path.exists(args.cost_out) else {"budgets": {}}
    cost_b = cost.get("budgets", {})

    # x-ось: среднее число объектов (если cost есть) иначе сам бюджет k.
    def xval(k: int) -> float:
        b = cost_b.get(str(k))
        return b["mean_objects"] if b else float(k)

    series: Dict[str, List] = {c: [] for c in conds}
    xs: Dict[str, List] = {c: [] for c in conds}
    for c in conds:
        for k in budgets:
            m = _find_metric(args.ckpt_root, k, c, eval_mode)
            if m is None or metric not in m:
                continue
            xs[c].append(xval(k))
            series[c].append(m[metric])

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in conds:
        if xs[c]:
            ax.plot(xs[c], series[c], marker="o", label=f"{c} ({metric})")
    ax.set_xlabel("mean rendered objects (edges)")
    ax.set_ylabel(metric)
    ax.set_title("Position scaling curve: quality vs #objects")
    ax.grid(True, alpha=0.3)

    # Вторая ось — стоимость в токенах (текст vs картинка).
    if cost_b:
        ax2 = ax.twinx()
        ks = sorted((int(k) for k in cost_b), key=int)
        xo = [cost_b[str(k)]["mean_objects"] for k in ks]
        txt = [cost_b[str(k)]["text_tokens"] for k in ks]
        img = [cost_b[str(k)]["image_tokens"] for k in ks]
        ax2.plot(xo, txt, ls="--", color="gray", alpha=0.7, label="text tokens")
        ax2.plot(xo, img, ls=":", color="gray", alpha=0.7, label="image tokens")
        ax2.set_ylabel("input tokens (cost)")
        ax2.legend(loc="lower right", fontsize=8)

    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig_out) or ".", exist_ok=True)
    fig.savefig(args.fig_out, dpi=150)
    print(f"[plot] фигура: {args.fig_out}")

    # CSV рядом.
    csv = args.fig_out.rsplit(".", 1)[0] + ".csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("condition,k,mean_objects,%s,text_tokens,image_tokens\n" % metric)
        for c in conds:
            for k in budgets:
                m = _find_metric(args.ckpt_root, k, c, eval_mode)
                b = cost_b.get(str(k), {})
                if m is None or metric not in m:
                    continue
                f.write(f"{c},{k},{b.get('mean_objects','')},{m[metric]},"
                        f"{b.get('text_tokens','')},{b.get('image_tokens','')}\n")
    print(f"[plot] таблица: {csv}")


# ────────────────────────────────────────────────────────────────
# selftest — синтетика без внешних зависимостей
# ────────────────────────────────────────────────────────────────
def run_selftest(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) стоимость токенов картинки (Qwen) для набора сторон
    print("[selftest] qwen visual tokens:")
    for side in (336, 448, 560, 672, 784):
        print(f"    side={side:>4} → {qwen_visual_tokens(side, side, args.qwen_min_pixels, args.qwen_max_pixels)} tok")

    # 2) count_objects на двух форматах
    trip = "(A, r1, B)\n(A, r2, C)\n(A, r3, [?])"
    facts = "- A --rel--> B\n- C --rel--> A"
    assert count_objects(trip) == 3 and count_objects(facts) == 2
    print(f"[selftest] count_objects: triples={count_objects(trip)} facts={count_objects(facts)} OK")

    # 3) синтетическая кривая: качество растёт и выходит на плато; токены линейны
    budgets = [1, 2, 4, 8, 16, 32]
    q_visual = [0.20, 0.28, 0.36, 0.41, 0.42, 0.40]
    q_text = [0.19, 0.24, 0.29, 0.32, 0.33, 0.33]
    q_text_feat = [0.20, 0.29, 0.37, 0.41, 0.42, 0.415]
    text_tok = [heuristic_tokens("x" * (12 * k)) for k in budgets]
    img_tok = qwen_visual_tokens(args.target_size, args.target_size,
                                 args.qwen_min_pixels, args.qwen_max_pixels)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, q in [("visual", q_visual), ("text", q_text), ("text+feat", q_text_feat)]:
        ax.plot(budgets, q, marker="o", label=f"{name} (MRR)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rendered objects (edges)")
    ax.set_ylabel("MRR")
    ax.set_title("[selftest] Position scaling curve (synthetic)")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(budgets, text_tok, ls="--", color="gray", label="text tokens")
    ax2.plot(budgets, [img_tok] * len(budgets), ls=":", color="gray", label=f"image tokens (Qwen@{args.target_size})")
    ax2.set_ylabel("input tokens (cost)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = args.fig_out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[selftest] фигура сохранена: {out}")


# ────────────────────────────────────────────────────────────────
def _parse_budgets(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["build", "cost", "plot", "all", "selftest"], default="selftest")
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--target_size", type=int, default=560, help="сторона рендера (Qwen: 448..751)")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--qwen_min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--qwen_max_pixels", type=int, default=720 * 28 * 28)
    # build
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--splits", default="stage2_test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_base", default=os.path.join(_ROOT, "graphvis_scaling"))
    # cost
    ap.add_argument("--split", default="stage2_test")
    ap.add_argument("--text_field", default="neighbors_text",
                    help="колонка с текстовой окрестностью (neighbors_text | prompt_triples | …)")
    ap.add_argument("--cost_out", default=os.path.join(_ROOT, "graphvis_scaling", "cost.json"))
    ap.add_argument("--limit", type=int, default=0)
    # plot
    ap.add_argument("--ckpt_root", default=os.path.join(_ROOT, "graphvis_ckpts_scaling"))
    ap.add_argument("--conditions", default="visual,text,text_feat")
    ap.add_argument("--eval_mode", choices=["rank", "sample", "auto"], default="auto",
                    help="режим метрик: rank|sample читает только свой паттерн; "
                         "auto — авто-детект, падает при смешении режимов")
    ap.add_argument("--metric", default="MRR")
    ap.add_argument("--fig_out", default=os.path.join(_ROOT, "graphvis_scaling", "scaling_curve.png"))
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.mode == "build":
        run_build(args)
    elif args.mode == "cost":
        run_cost(args)
    elif args.mode == "plot":
        run_plot(args)
    elif args.mode == "all":
        run_build(args); run_cost(args); run_plot(args)
    else:
        run_selftest(args)


if __name__ == "__main__":
    main()
