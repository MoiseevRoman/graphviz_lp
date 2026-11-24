"""Зонд чтения на картинках ОСНОВНОЙ статьи (1-хоп звезда, graphvis_scaling/k<K>).

label_read_probe мерил чтение на path-рендерах, legibility_probe — на ранних
рендерах из Neo4j. Здесь — ровно те PNG, на которых обучались/оценивались
visual-условия статьи (560 px, 400 виз. токенов), без перерендера и без Neo4j.

Без обучения: базовая Qwen2.5-VL, две задачи на картинку:
    names    — перечислить имена всех сущностей в боксах, кроме головы
               (видит ли модель подписи вообще);
    triples  — для каждой стрелки у головы вывести «relation | entity»
               (привязка подписи ребра к узлу — то, что нужно link prediction;
               по обзору «When Vision Meets Graphs» узкое место именно здесь).

Эталон = то, что реально нарисовано: neighbors_text → тот же отбор, что в рендере
(visualizer.compute_visible_subgraph: первые max_nodes_display узлов), подписи
через format_label (обрезка до 30 символов, '_' → ' '). Направление стрелки не
оценивается. Запросы — те же seed-42 индексы, что в rank-eval (общий тест).

    python scripts/star_read_probe.py --ks 4,16 --n 60
    python scripts/star_read_probe.py --ks 16 --n 60 --quants nf4,vision_bf16
    # дообученный адаптер (напр. после stage1) — те же вопросы, свой ключ в json:
    python scripts/star_read_probe.py --ks 16 --quants vision_bf16 \
        --adapter graphvis_ckpts_s1/k16/ctx_visual_s1/adapter_stage1 --label s1

Задачи s1_nodes / s1_triples — дословные формулировки stage1
(build_stage1_from_stage2: «List all nodes shown in the graph.» / «List all triples
in the graph.»), чтобы адаптер после stage1 спрашивали в его формате. Скоринг
не зависит от формата ответа: имя/пара засчитаны, если нормализованный эталон
встречается в ответе (в одной строке или одной скобке «(…)» для пар).
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
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _p in (_ROOT, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if "graphvis_lp" not in sys.modules:
    try:
        if os.path.dirname(_ROOT) not in sys.path:
            sys.path.insert(0, os.path.dirname(_ROOT))
        __import__("graphvis_lp")
    except Exception:
        _pkg = types.ModuleType("graphvis_lp")
        _pkg.__path__ = [_ROOT]
        sys.modules["graphvis_lp"] = _pkg

PROMPTS = {
    "names": ("The image shows a knowledge graph centered on '{head}'. List every entity "
              "name written in the boxes, except '{head}' and '[?]', exactly as written, "
              "one per line. Output only the names."),
    "s1_nodes": "List all nodes shown in the graph.",
    "s1_triples": "List all triples in the graph.",
    "triples": ("The image shows a knowledge graph centered on '{head}'. For every solid "
                "arrow connecting '{head}' to another box, write one line in the format: "
                "relation | entity\nwhere relation is the label written on the arrow and "
                "entity is the name in the box at its other end. Skip the dashed arrow to "
                "'[?]'. Output only these lines."),
}
_SIM = 0.8


def _norm(s: str) -> str:
    s = s.lower().replace("…", "").replace("...", "")
    s = re.sub(r"^[\s\-\*\d\.\)]+", "", s)          # маркеры списка "1. ", "- "
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sim(a: str, b: str) -> bool:
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= _SIM


def _units(output: str) -> list:
    """Единицы ответа: «(…)»-группы, если они есть (формат stage1-троек), иначе
    строки; для «a, b, c» в одну строку (формат stage1-узлов) — ещё и по запятым.
    Рёбра к заглушке [?] на картинке есть, но в эталон не входят — их отбрасываем."""
    groups = re.findall(r"\(([^()]*)\)", output)
    if len(groups) >= 2:
        return [_norm(g) for g in groups if "?" not in g and _norm(g)]
    out = []
    for ln in output.splitlines():
        if "|" in ln or "-->" in ln:
            if "[?]" not in ln and _norm(ln):
                out.append(_norm(ln))
            continue
        out += [_norm(x) for x in re.split(r"[;,]", ln) if _norm(x)]
    return out


def _has(unit: str, gold: str) -> bool:
    return bool(gold) and (gold in unit or _sim(gold, unit))


def score_names(gold_names, output: str, head: str = "") -> dict:
    full = " " + _norm(output) + " "
    # голову модель часто перечисляет тоже (формат stage1) — это не ошибка и не попадание;
    # без исключения «CeCe Winans» fuzzy-матчился бы с соседом «BeBe Winans»
    units = [u for u in _units(output) if u != _norm(head)]
    gold = [_norm(x) for x in gold_names]
    return {"n_gold": len(gold), "n_out": len(units),
            "exact": sum(f" {g} " in full for g in gold),
            "fuzzy": sum(any(_has(u, g) for u in units) for g in gold),
            "precise": sum(any(_has(u, g) or (len(u) > 3 and u in g) for g in gold)
                           for u in units)}


def score_triples(gold_pairs, output: str) -> dict:
    """gold_pairs: [(relation, entity)]. Пара засчитана, если в ОДНОЙ единице ответа
    (строка или «(…)») есть и подпись отношения, и имя сущности; entity — только имя."""
    units = _units(output)
    gold = [(_norm(r), _norm(e)) for r, e in gold_pairs]

    def _ent(u, ge):   # имя без учёта отношения: подстрока или fuzzy к хвосту строки
        w = u.split()
        return ge in u or any(_sim(ge, " ".join(w[i:])) for i in range(min(4, len(w))))

    def _pair(u, gr, ge):
        if f" {gr} " not in f" {u} ":
            return False
        rest = f" {u} ".replace(f" {gr} ", " ", 1).strip()   # строка без подписи отношения
        return ge in u or _sim(ge, rest)
    return {"n_gold": len(gold), "n_out": len(units),
            "pair": sum(any(_pair(u, gr, ge) for u in units) for gr, ge in gold),
            "entity": sum(any(_ent(u, ge) for u in units) for _, ge in gold),
            "precise": sum(any(_pair(u, gr, ge) for gr, ge in gold) for u in units)}


def build_items(k: int, n: int, data_base: str, max_nodes: int = 0) -> list:
    from datasets import load_from_disk
    from add_text_encodings import parse_neighbors_text
    from relayout_images import _build_sg
    from config import VisualizationConfig
    from visualizer import compute_visible_subgraph, format_label

    ds = load_from_disk(os.path.join(_ROOT, data_base, f"k{k}", "datasets", "stage2_test"))
    random.seed(42)   # те же индексы, что в rank-eval (eval_n=1000 из первых 2000)
    idx = sorted(random.sample(range(min(len(ds), 2000)), 1000))
    idx = idx[::max(1, len(idx) // n)][:n]
    vcfg = VisualizationConfig()
    if max_nodes > 0:     # рендер с --max_nodes (K>12): эталон = то, что реально нарисовано
        vcfg.max_nodes_display = max_nodes
        vcfg.max_edges_display = max(vcfg.max_edges_display, max_nodes + 10)
    L = vcfg.max_label_len
    items = []
    for i in idx:
        r = ds[i]
        edges, order = parse_neighbors_text(r["neighbors_text"])
        if not edges:
            continue
        sg = _build_sg(r["head"], r["relation"], edges, order, "")
        nodes, vis_edges = compute_visible_subgraph(sg, vcfg)
        head = r["head"]
        names = [format_label(nd["name"], L) for nd in nodes if nd["name"] != head]
        pairs = []
        for e in vis_edges:
            other = e["target"] if e["source"] == head else e["source"]
            if head not in (e["source"], e["target"]):
                continue
            pairs.append((format_label(e["relation"], L), format_label(other, L)))
        img = r["image_path"]
        if not os.path.isabs(img):
            img = os.path.join(_ROOT, img)
        items.append({"idx": i, "query_id": r["query_id"], "head": format_label(head, L),
                      "names": names, "pairs": pairs, "image": img})
    return items


def run(args) -> None:
    import torch
    from PIL import Image
    from graphvis_lp.config import TrainConfig
    from graphvis_lp.training.families import load_model_and_processor
    from graphvis_lp.training.model import quant_config, vision_precision
    tag = args.label or ("base" if not args.adapter else os.path.basename(args.adapter.rstrip("/\\")))

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    quants = [x.strip() for x in args.quants.split(",") if x.strip()]
    items_by_k = {k: build_items(k, args.n, args.data_base) for k in ks}
    for k, its in items_by_k.items():
        miss = [it["image"] for it in its if not os.path.exists(it["image"])]
        print(f"K={k}: {len(its)} images, mean drawn neighbors "
              f"{sum(len(it['names']) for it in its) / max(len(its), 1):.1f}, missing {len(miss)}",
              flush=True)
        if miss:
            raise SystemExit(f"нет картинок, напр. {miss[0]}")

    results = {"n": args.n, "summary": {}, "details": []}
    if os.path.exists(args.out):   # дозапись: уже посчитанные варианты пропускаем
        results = json.load(open(args.out, encoding="utf-8"))

    for q in quants:
        todo = [(k, t) for k in ks for t in tasks
                if f"{tag}/{q}/K{k}/{t}" not in results["summary"]]
        if not todo:
            print(f"[{q}] всё уже посчитано — пропуск", flush=True)
            continue
        cfg = TrainConfig()
        cfg.model_name = args.model
        cfg.model_family = "auto"
        cfg.vision_bf16 = (q == "vision_bf16")
        load_kwargs = {"device_map": "cuda:0"}
        qc = quant_config(cfg)
        if qc is not None:
            load_kwargs["quantization_config"] = qc
        model, processor = load_model_and_processor(cfg, load_kwargs, torch.bfloat16)
        n4, nv = vision_precision(model)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
            print(f"[{tag}] adapter: {args.adapter}", flush=True)
        model.eval()
        print(f"[{q}] vision Linear in 4-bit: {n4}/{nv}", flush=True)
        if q == "vision_bf16" and n4:
            raise SystemExit("vision_bf16 не сработал (зрительная башня квантована)")

        for k, t in todo:
            t0 = time.time()
            tot, vis_tok = {}, []
            for it in items_by_k[k]:
                img = Image.open(it["image"]).convert("RGB")
                prompt = PROMPTS[t].format(head=it["head"])
                msgs = [{"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
                if "image_grid_thw" in enc:
                    vis_tok.append(int(enc["image_grid_thw"].prod()) // 4)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
                ans = processor.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)[0]
                s = (score_names(it["names"], ans, it["head"]) if t in ("names", "s1_nodes")
                     else score_triples(it["pairs"], ans))
                for key, v in s.items():
                    tot[key] = tot.get(key, 0) + v
                results["details"].append({"model": tag, "quant": q, "k": k, "task": t,
                                           "idx": it["idx"],
                                           **s, "output": ans})
            g, o = max(tot["n_gold"], 1), max(tot["n_out"], 1)
            if t in ("names", "s1_nodes"):
                summ = {"recall_exact_%": round(100 * tot["exact"] / g, 1),
                        "recall_fuzzy_%": round(100 * tot["fuzzy"] / g, 1),
                        "precision_%": round(100 * tot["precise"] / o, 1)}
            else:
                summ = {"pair_recall_%": round(100 * tot["pair"] / g, 1),
                        "entity_recall_%": round(100 * tot["entity"] / g, 1),
                        "pair_precision_%": round(100 * tot["precise"] / o, 1)}
            summ.update({"gold_per_img": round(tot["n_gold"] / len(items_by_k[k]), 1),
                         "out_per_img": round(tot["n_out"] / len(items_by_k[k]), 1),
                         "mean_visual_tokens": round(sum(vis_tok) / max(len(vis_tok), 1)),
                         "sec": round(time.time() - t0)})
            results["summary"][f"{tag}/{q}/K{k}/{t}"] = summ
            print(f"[{tag}/{q}/K{k}/{t}] {json.dumps(summ)}", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
        del model
        torch.cuda.empty_cache()

    print(f"\n=== reading the paper's 1-hop renders ({args.n} images per K) ===")
    for key, s in results["summary"].items():
        print(f"{key:40s} {json.dumps(s)}")
    print(f"saved: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="4,16")
    ap.add_argument("--n", type=int, default=60, help="картинок на K")
    ap.add_argument("--tasks", default="names,triples,s1_nodes,s1_triples")
    ap.add_argument("--adapter", default="", help="PEFT-адаптер поверх базы (пусто = базовая модель)")
    ap.add_argument("--label", default="", help="ключ модели в json (по умолчанию base / имя папки адаптера)")
    ap.add_argument("--quants", default="nf4", help="nf4 (как в статье) и/или vision_bf16")
    ap.add_argument("--data_base", default="graphvis_scaling")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--out", default="graphvis_ckpts_scaling_common/star_read_probe.json")
    ap.add_argument("--dry_run", action="store_true", help="только эталоны, без модели")
    args = ap.parse_args()
    if args.dry_run:
        for k in [int(x) for x in args.ks.split(",")]:
            its = build_items(k, args.n, args.data_base)
            it = its[0]
            print(f"K={k}: {len(its)} items; e.g. {it['head']} -> {it['names'][:5]} | {it['pairs'][:3]}")
            print("  ", it["image"], os.path.exists(it["image"]))
        return
    run(args)


if __name__ == "__main__":
    main()
