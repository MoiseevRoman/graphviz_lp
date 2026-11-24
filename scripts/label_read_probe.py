"""Проверка ЧТЕНИЯ: может ли VLM вообще прочитать имена кандидатов с рендера?

Без обучения: базовая Qwen2.5-VL получает картинку path-подграфа (как в
graphvis_paths_iso) и просьбу перечислить имена в жёлтых плашках. Считаем долю
правильно прочитанных имён. Варианты:
    render:  base   — как в датасете (560 px, шрифт ×1, 400 виз. токенов)
             font   — 560 px, шрифт ×2.5 (те же 400 токенов)
             hires  — 1120 px, шрифт ×1 (~1600 токенов; верхняя граница по разрешению)
    quant:   nf4         — вся модель в 4 бит (как во всех прогонах)
             vision_bf16 — LM в 4 бит, зрительная башня в bf16

Читаем так: если при vision_bf16 / font доля прочитанных имён резко выше, чем
nf4 / base, то картинка в основных прогонах проигрывала из-за читаемости, и есть
смысл переобучить visual_paths с этим режимом. Если доля и так высокая или не
растёт — дело не в разрешении/квантовании.

    python scripts/label_read_probe.py prepare            # рендер, без GPU
    python scripts/label_read_probe.py run --n 60         # GPU
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import time
import types

for _stream in (sys.stdout, sys.stderr):      # cp1251-консоль Windows
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# репозиторий импортируемым как `graphvis_lp` независимо от имени папки (как в viability_gate)
if "graphvis_lp" not in sys.modules:
    try:
        if os.path.dirname(_ROOT) not in sys.path:
            sys.path.insert(0, os.path.dirname(_ROOT))
        __import__("graphvis_lp")
    except Exception:
        _pkg = types.ModuleType("graphvis_lp")
        _pkg.__path__ = [_ROOT]
        sys.modules["graphvis_lp"] = _pkg

RENDERS = {"base": (560, 1.0), "font": (560, 2.5), "hires": (1120, 1.0)}
PROMPT = ("The image shows a graph. List every entity name written in the yellow boxes "
          "on the right side of the image, exactly as written, one per line. "
          "Output only the names.")


def _norm(s: str) -> str:
    s = s.lower().replace("…", "").replace("...", "")
    s = re.sub(r"^[\s\-\*\d\.\)]+", "", s)          # маркеры списка "1. ", "- "
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def score(labels, output: str) -> dict:
    lines = [_norm(x) for x in output.splitlines()]
    lines = [x for x in lines if x]
    gold = [_norm(x) for x in labels]
    exact = sum(g in lines for g in gold)
    fuzzy = sum(any(difflib.SequenceMatcher(None, g, x).ratio() >= 0.8 for x in lines)
                for g in gold)
    prec = sum(any(difflib.SequenceMatcher(None, g, x).ratio() >= 0.8 for g in gold)
               for x in lines)
    return {"n_gold": len(gold), "n_out": len(lines), "exact": exact, "fuzzy": fuzzy,
            "precise_lines": prec}


def prepare(args) -> None:
    from datasets import load_from_disk
    from path_retriever import PathGraph, PathSubgraphConfig, render
    from visualizer import format_label

    ds = load_from_disk(os.path.join(_ROOT, args.dataset_root, "stage2_test"))
    # те же запросы, что в rank-eval (seed 42), первые n_max из них
    random.seed(42)
    idx = sorted(random.sample(range(min(len(ds), 2000)), 1000))[::max(1, 1000 // args.n_max)][:args.n_max]
    g = PathGraph(os.path.join(_ROOT, args.data_dir), split="test")
    cfg = PathSubgraphConfig(n_show=20, max_paths_per_cand=5, preselect="cn_freq",
                             label_edges=False)
    items = []
    for k, i in enumerate(idx):
        r = ds[i]
        sg = g.build(r["head"], r["relation"], r["tail"], cfg)
        labels = [format_label(c, 30) for c in sg.shown]
        imgs = {}
        for name, (size, fs) in RENDERS.items():
            rel = f"{args.out_dir}/{name}/lp_{r['query_id']}.png"
            full = os.path.join(_ROOT, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            if not os.path.exists(full):
                render(sg, cfg, full, target_size=size, font_scale=fs)
            imgs[name] = rel
        items.append({"idx": i, "query_id": r["query_id"], "head": r["head"],
                      "labels": labels, "images": imgs})
        if (k + 1) % 20 == 0:
            print(f"prepared {k + 1}/{len(idx)}", flush=True)
    with open(os.path.join(_ROOT, args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    print(f"manifest: {len(items)} queries x {len(RENDERS)} renders")


def run(args) -> None:
    import torch
    from PIL import Image
    from graphvis_lp.config import TrainConfig
    from graphvis_lp.training.families import load_model_and_processor
    from graphvis_lp.training.model import quant_config, vision_precision

    items = json.load(open(os.path.join(_ROOT, args.out_dir, "manifest.json"), encoding="utf-8"))
    items = items[:args.n]
    renders = [x.strip() for x in args.renders.split(",") if x.strip()]
    quants = [x.strip() for x in args.quants.split(",") if x.strip()]
    results = {"n": len(items), "summary": {}, "details": []}

    for q in quants:
        cfg = TrainConfig()
        cfg.model_name = args.model
        cfg.model_family = "auto"
        cfg.vision_bf16 = (q == "vision_bf16")
        cfg.qwen_max_pixels = 1600 * 28 * 28        # чтобы hires (1120 px) не ужимался
        load_kwargs = {"device_map": "cuda:0"}
        qc = quant_config(cfg)
        if qc is not None:
            load_kwargs["quantization_config"] = qc
        model, processor = load_model_and_processor(cfg, load_kwargs, torch.bfloat16)
        model.eval()
        n4, nv = vision_precision(model)
        print(f"[{q}] vision Linear in 4-bit: {n4}/{nv}", flush=True)
        if q == "vision_bf16" and n4:
            raise SystemExit("vision_bf16 не сработал (зрительная башня квантована)")

        for rname in renders:
            t0 = time.time()
            tot = {"n_gold": 0, "exact": 0, "fuzzy": 0, "n_out": 0, "precise_lines": 0}
            vis_tok = []
            for it in items:
                img = Image.open(os.path.join(_ROOT, it["images"][rname])).convert("RGB")
                msgs = [{"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": PROMPT}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
                if "image_grid_thw" in enc:
                    vis_tok.append(int(enc["image_grid_thw"].prod()) // 4)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
                ans = processor.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)[0]
                s = score(it["labels"], ans)
                for k in tot:
                    tot[k] += s[k]
                results["details"].append({"quant": q, "render": rname, "idx": it["idx"],
                                           **s, "output": ans})
            summ = {"recall_exact_%": round(100 * tot["exact"] / tot["n_gold"], 1),
                    "recall_fuzzy_%": round(100 * tot["fuzzy"] / tot["n_gold"], 1),
                    "precision_%": round(100 * tot["precise_lines"] / max(tot["n_out"], 1), 1),
                    "mean_lines": round(tot["n_out"] / len(items), 1),
                    "mean_visual_tokens": round(sum(vis_tok) / max(len(vis_tok), 1)),
                    "sec": round(time.time() - t0)}
            results["summary"][f"{q}/{rname}"] = summ
            print(f"[{q}/{rname}] {json.dumps(summ)}", flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
        del model
        torch.cuda.empty_cache()

    print("\n=== label reading (20 candidate names per image, "
          f"{len(items)} images) ===")
    print(f"{'quant/render':24s} {'vis.tok':>7s} {'exact%':>7s} {'fuzzy%':>7s} {'prec%':>6s} {'lines':>6s}")
    for k, s in results["summary"].items():
        print(f"{k:24s} {s['mean_visual_tokens']:7d} {s['recall_exact_%']:7.1f} "
              f"{s['recall_fuzzy_%']:7.1f} {s['precision_%']:6.1f} {s['mean_lines']:6.1f}")
    print(f"saved: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("--dataset_root", default="graphvis_paths_iso/datasets")
    ap.add_argument("--data_dir", default="FB15k-237")
    ap.add_argument("--out_dir", default="graphvis_paths_iso/label_probe")
    ap.add_argument("--n_max", type=int, default=100, help="prepare: сколько запросов рендерить")
    ap.add_argument("--n", type=int, default=60, help="run: сколько картинок на вариант")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--renders", default="base,font,hires")
    ap.add_argument("--quants", default="nf4,vision_bf16")
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--out", default="graphvis_ckpts_paths_iso/label_probe.json")
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else run(args)


if __name__ == "__main__":
    main()
