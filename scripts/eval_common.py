#!/usr/bin/env python3
"""Переоценка всех обученных чекпойнтов scaling на ОБЩЕМ тест-сете.

Идея: обучение шло под iso-token бюджет (равные ресурсы на train — это ок),
но тест каждого условия шёл на своём подмножестве (разный eval_n) → MRR
несопоставимы. Здесь мы прогоняем eval-only с ОДИНАКОВЫМ eval_n для всех
условий каждого K, поэтому оцениваются одни и те же тестовые тройки
(датасет на K общий, тест = детерминированный префикс stage2_test).

Ничего не удаляем и не переобучаем: адаптеры грузятся из старого дерева
(--ckpt_root, по умолчанию graphvis_ckpts_scaling), а метрики пишутся в новое
(--out_root, по умолчанию graphvis_ckpts_scaling_common).

Пример:
    python scripts/eval_common.py --budgets 1,4,8,16 \
        --conditions text,text_feat,visual --eval_n 1000 --eval_mode rank
"""
import argparse
import os
import subprocess
import sys
from typing import List

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
_PY = sys.executable


def _run(cmd: List[str], dry: bool) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="1,4,8,16", help="значения K через запятую")
    ap.add_argument("--conditions", default="text,text_feat,visual")
    ap.add_argument("--eval_n", type=int, default=1000,
                    help="ОДИН и тот же eval_n для всех условий → общий тест-сет")
    ap.add_argument("--eval_mode", choices=["rank", "generative", "sample"], default="rank")
    ap.add_argument("--num_samples", type=int, default=300)
    ap.add_argument("--gen_batch", type=int, default=50)
    ap.add_argument("--cand_cap", type=int, default=50)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--data_dir", default=os.path.join(_ROOT, "FB15k-237"))
    ap.add_argument("--data_base", default=os.path.join(_ROOT, "graphvis_scaling"),
                    help="где лежат датасеты per-K (…/k{K}/datasets)")
    ap.add_argument("--ckpt_root", default=os.path.join(_ROOT, "graphvis_ckpts_scaling"),
                    help="откуда брать обученные адаптеры (не трогается)")
    ap.add_argument("--out_root", default=os.path.join(_ROOT, "graphvis_ckpts_scaling_common"),
                    help="куда писать новые (сопоставимые) метрики")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    Ks = [int(x) for x in args.budgets.split(",") if x.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    dry = args.dry_run

    print(f"Общий тест-сет: eval_n={args.eval_n} (одинаков для всех условий), "
          f"eval_mode={args.eval_mode}")
    print(f"адаптеры ← {args.ckpt_root}")
    print(f"метрики → {args.out_root}\n")

    missing = []
    for k in Ks:
        ds_root = os.path.join(args.data_base, f"k{k}", "datasets")
        for cond in conditions:
            src = os.path.join(args.ckpt_root, f"k{k}", f"ctx_{cond}")
            adapter = os.path.join(src, "adapter")
            outdir = os.path.join(args.out_root, f"k{k}", f"ctx_{cond}")
            if not (dry or os.path.isdir(adapter)):
                print(f"[k={k}] «{cond}»: НЕТ адаптера в {adapter} — пропуск")
                missing.append((k, cond))
                continue
            os.makedirs(outdir, exist_ok=True) if not dry else None
            print(f"[k={k}] «{cond}» eval-only → {outdir}")
            cmd = [_PY, os.path.join(_SCRIPTS, "viability_gate.py"),
                   "--eval_only",
                   "--adapter_dir", adapter,
                   "--model", args.model,
                   "--dataset_root", ds_root,
                   "--data_dir", args.data_dir,
                   "--output_dir", outdir,
                   "--out", os.path.join(outdir, "report.json"),
                   "--eval_mode", args.eval_mode,
                   "--context_mode", cond,
                   "--eval_n", str(args.eval_n),
                   "--cand_cap", str(args.cand_cap),
                   "--num_samples", str(args.num_samples),
                   "--gen_batch", str(args.gen_batch)]
            _run(cmd, dry)

    if missing:
        print(f"\nПропущено (нет адаптера): {missing}")
    print("\nГотово. Пересобрать кривую: python scripts/eval_common.py уже писал метрики; "
          "для сводки используйте свой построитель по --out_root.")


if __name__ == "__main__":
    main()
