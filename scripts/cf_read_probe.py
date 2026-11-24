"""Контрфактическая проба чтения: одна и та же картинка с минимальной правкой графа.

Контроль «чужая картинка» (read_choice_probe) меняет всё сразу — голову, раскладку,
имена. Здесь меняется ОДИН элемент графа, а негативы — настоящие соседи головы в KG,
поэтому ответить «по памяти» нельзя: различить их можно только глядя на картинку.

Варианты рендера (K=16, те же seed-42 тестовые запросы, рендер в одном окружении):
    orig    исходный подграф (перерисован тем же graphviz, что и правки);
    drop    ~30 % соседей вместе с рёбрами невидимы (место в раскладке сохранено);
    swap    у двух рёбер голова→сосед переставлены подписи (узлы и раскладка те же);
    detach  у 1–2 соседей стрелка к голове невидима, бокс на прежнем месте.

Задачи (Да/Нет, logit(Yes) − logit(No), AUC; позитив/негатив — по ПРАВЛЕННОЙ картинке):
    drop    «есть ли бокс X»: оставшиеся (1) против удалённых (0);
    swap    «стрелка r между головой и X»: новые пары (1) против исходных пар (0);
    detach  «соединён ли X с головой стрелкой»: присоединённые (1) против отцепленных (0).
Те же вопросы задаются и на orig (там истина другая). Чтение = AUC(правка) − AUC(orig),
95 % ДИ бутстрепом по картинкам. Модель, отвечающая по памяти KG, даёт одинаковые
ответы на обе картинки → d ≈ 0; идеальный читатель: drop/detach d ≈ +0.5, swap ≈ +1.

    python scripts/cf_read_probe.py build                 # рендеры + вопросы (без GPU)
    python scripts/cf_read_probe.py run --label base
    python scripts/cf_read_probe.py run --label lp_vbf16 --adapter graphvis_ckpts_s1/k16/ctx_visual_vbf16/adapter
    python scripts/cf_read_probe.py report
Также `build` пишет LP-подвыборки для контрфактического rank-eval:
    graphvis_cf/k16/datasets/{orig300,drop300}/stage2_test (300 запросов; в drop300
    удалённые соседи убраны И с картинки, И из neighbors_text).
"""

from __future__ import annotations

import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):   # Windows cp1251: печать «→»/«Δ» роняла скрипт
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import os
import random
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)
import read_choice_probe as rcp   # noqa: E402
import star_read_probe as srp     # noqa: E402

OUT_BASE = os.path.join(_ROOT, "graphvis_cf", "k16")
VARIANTS = ("orig", "drop", "swap", "detach")
TASKS = ("drop", "swap", "detach")
Q_CONN = ("The image shows a knowledge graph centered on '{head}'. "
          "Is '{ent}' connected to '{head}' by an arrow? Answer Yes or No.")


def _visible(row, vcfg):
    from add_text_encodings import parse_neighbors_text
    from relayout_images import _build_sg
    from visualizer import compute_visible_subgraph
    edges, order = parse_neighbors_text(row["neighbors_text"])
    if not edges:
        return None
    sg = _build_sg(row["head"], row["relation"], edges, order, "")
    nodes, vis_edges = compute_visible_subgraph(sg, vcfg)
    return sg, nodes, vis_edges


def _render(sg, name, variant, vcfg_size):
    from config import VisualizationConfig
    from visualizer import visualize_task
    cfg = VisualizationConfig()
    cfg.output_dir = os.path.join(OUT_BASE, variant)
    cfg.target_size = vcfg_size
    cfg.cache_enabled = False
    p = visualize_task(sg, name, sg["rel"], cfg)
    return os.path.relpath(os.path.abspath(p), _ROOT).replace("\\", "/")


def make_variants(row, rng, vcfg, drop_frac):
    """Правки ВИДИМОГО подграфа. Возвращает dict variant → (sg, meta) или None."""
    from visualizer import format_label
    vis = _visible(row, vcfg)
    if vis is None:
        return None
    sg, nodes, edges = vis
    head, L = row["head"], vcfg.max_label_len
    lab = lambda x: format_label(x, L)  # noqa: E731
    nbrs = [n["name"] for n in nodes if n["name"] != head]
    # только соседи с уникальной подписью бокса (иначе «есть ли X» неоднозначно)
    cnt = {}
    for n in nbrs:
        cnt[lab(n)] = cnt.get(lab(n), 0) + 1
    nbrs = [n for n in nbrs if cnt[lab(n)] == 1 and srp._norm(lab(n)) != srp._norm(lab(head))]
    deg = {n: sum(1 for e in edges if n in (e["source"], e["target"])) for n in nbrs}
    if len(nbrs) < 4:
        return None
    base = {"head": head, "rel": sg["rel"], "head_description": "", "stats": {},
            "nodes": [dict(n) for n in nodes], "edges": [dict(e) for e in edges]}

    def clone():
        return json.loads(json.dumps(base))

    out = {"orig": (clone(), {})}

    k = max(1, int(round(drop_frac * len(nbrs))))
    dropped = rng.sample(nbrs, k)
    g = clone()
    # невидимы, но занимают место → раскладка и масштаб остальных не меняются
    for n in g["nodes"]:
        n["hidden"] = n["name"] in dropped
    for e in g["edges"]:
        e["hidden"] = e["source"] in dropped or e["target"] in dropped
    kept = [n for n in nbrs if n not in dropped]
    out["drop"] = (g, {"dropped": dropped, "kept": kept})

    single = [n for n in nbrs if deg[n] == 1]
    rel_of = {}
    for e in edges:
        o = e["target"] if e["source"] == head else e["source"]
        if o in single:
            rel_of[o] = e["relation"]
    cand = [n for n in single if n in rel_of]
    rng.shuffle(cand)
    swaps, used = [], set()
    for a in cand:
        if len(swaps) == 2 or a in used:
            continue
        for b in cand:
            if b != a and b not in used and srp._norm(lab(rel_of[a])) != srp._norm(lab(rel_of[b])):
                swaps.append((a, b))
                used |= {a, b}
                break
    if swaps:
        g = clone()
        new_rel = {}
        for a, b in swaps:
            new_rel[a], new_rel[b] = rel_of[b], rel_of[a]
        for e in g["edges"]:
            o = e["target"] if e["source"] == head else e["source"]
            if o in new_rel:
                e["relation"] = new_rel[o]
        out["swap"] = (g, {"pairs": [[a, lab(rel_of[a]), b, lab(rel_of[b])] for a, b in swaps]})

    det_pool = [n for n in single if n not in used] or single
    if len(single) >= 3:
        det = rng.sample(det_pool, min(2, len(det_pool)))
        g = clone()
        for e in g["edges"]:
            e["hidden"] = e["source"] in det or e["target"] in det
        attached = [n for n in nbrs if n not in det]
        out["detach"] = (g, {"detached": det, "attached": attached})
    return out


def build_questions(item, rng, per=4):
    """Вопросы с меткой по правленой картинке; каждый задаётся и на orig."""
    from config import VisualizationConfig
    from visualizer import format_label
    L = VisualizationConfig().max_label_len
    lab = lambda x: format_label(x, L)  # noqa: E731
    h = lab(item["head"])
    qs = []
    m = item["meta"]
    if "drop" in m:
        neg = m["drop"]["dropped"][:per]
        pos = rng.sample(m["drop"]["kept"], min(per, len(m["drop"]["kept"])))
        qs += [{"task": "drop", "label": 1, "text": rcp.Q_NODE.format(head=h, ent=lab(x))} for x in pos]
        qs += [{"task": "drop", "label": 0, "text": rcp.Q_NODE.format(head=h, ent=lab(x))} for x in neg]
    if "swap" in m:
        for a, ra, b, rb in m["swap"]["pairs"]:
            A, B = lab(a), lab(b)
            qs += [{"task": "swap", "label": 1, "text": rcp.Q_EDGE.format(head=h, rel=rb, ent=A)},
                   {"task": "swap", "label": 1, "text": rcp.Q_EDGE.format(head=h, rel=ra, ent=B)},
                   {"task": "swap", "label": 0, "text": rcp.Q_EDGE.format(head=h, rel=ra, ent=A)},
                   {"task": "swap", "label": 0, "text": rcp.Q_EDGE.format(head=h, rel=rb, ent=B)}]
    if "detach" in m:
        neg = m["detach"]["detached"]
        pos = rng.sample(m["detach"]["attached"], min(per, len(m["detach"]["attached"])))
        qs += [{"task": "detach", "label": 1, "text": Q_CONN.format(head=h, ent=lab(x))} for x in pos]
        qs += [{"task": "detach", "label": 0, "text": Q_CONN.format(head=h, ent=lab(x))} for x in neg]
        # контроль присутствия: бокс отцепленного узла на картинке ЕСТЬ
        qs += [{"task": "detach_box", "label": 1, "text": rcp.Q_NODE.format(head=h, ent=lab(x))} for x in neg]
    return qs


def eval_indices(n_test=2000, n=1000):
    random.seed(42)   # как в rank-eval / star_read_probe
    return sorted(random.sample(range(n_test), n))


def cmd_build(args):
    from datasets import load_from_disk
    from config import VisualizationConfig
    vcfg = VisualizationConfig()
    ds = load_from_disk(os.path.join(_ROOT, args.data_base, "k16", "datasets", "stage2_test"))
    idx = eval_indices(min(len(ds), 2000))
    items = []
    # ── проба: n картинок равномерно по eval-индексам ──
    for i in idx[::max(1, len(idx) // args.n)]:
        if len(items) >= args.n:
            break
        r = ds[i]
        rng = random.Random(1000 + i)
        var = make_variants(r, rng, vcfg, args.drop_frac)
        if var is None or not {"swap", "detach"} <= set(var):
            continue
        it = {"idx": i, "query_id": r["query_id"], "head": r["head"],
              "orig_image": r["image_path"], "images": {}, "meta": {}}
        for v, (g, meta) in var.items():
            it["images"][v] = _render(g, f"{r['query_id']}_{v}", v, args.size)
            if meta:
                it["meta"][v] = meta
        it["questions"] = build_questions(it, random.Random(2000 + i))
        items.append(it)
    with open(os.path.join(OUT_BASE, "probe_items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    nq = {t: sum(q["task"] == t for it in items for q in it["questions"]) for t in TASKS + ("detach_box",)}
    print(f"probe: {len(items)} картинок, вопросов {nq}")

    # ── LP: orig300 / drop300 (те же запросы, правка и в картинке, и в тексте) ──
    rows_o, rows_d = [], []
    for i in idx[:args.n_lp]:
        r = dict(ds[i])
        rng = random.Random(3000 + i)
        vis = _visible(r, vcfg)
        if vis is None:
            continue
        var = make_variants(r, rng, vcfg, args.drop_frac)
        if var is None:
            continue
        dropped = var["drop"][1]["dropped"]
        o = dict(r, orig_idx=i, dropped=json.dumps([], ensure_ascii=False),
                 image_path=_render(var["orig"][0], f"{r['query_id']}_lp", "lp_orig", args.size))
        from add_text_encodings import parse_neighbors_text
        keep = []
        for ln in r["neighbors_text"].split("\n"):
            e, _ = parse_neighbors_text(ln)
            if e and (e[0][0] in dropped or e[0][2] in dropped):
                continue
            keep.append(ln)
        d = dict(r, orig_idx=i, dropped=json.dumps(dropped, ensure_ascii=False),
                 image_path=_render(var["drop"][0], f"{r['query_id']}_lp", "lp_drop", args.size),
                 neighbors_text="\n".join(keep))
        rows_o.append(o)
        rows_d.append(d)
    # Dataset.from_dict падает на py3.14+dill (хеширование) → select + map с явным fingerprint
    sel = [row["orig_idx"] for row in rows_o]
    for name, rows in (("orig300", rows_o), ("drop300", rows_d)):
        upd = [{k: row[k] for k in ("image_path", "neighbors_text", "orig_idx", "dropped")} for row in rows]
        sub_ds = ds.select(sel, new_fingerprint=f"cf_sel_{name}")
        sub_ds = sub_ds.map(lambda _, j: upd[j], with_indices=True, keep_in_memory=True,
                            new_fingerprint=f"cf_upd_{name}")
        p = os.path.join(OUT_BASE, "datasets", name, "stage2_test")
        sub_ds.save_to_disk(p)
        print(f"LP {name}: {len(rows)} запросов → {os.path.relpath(p, _ROOT)}")
    # viability_gate грузит stage2_train даже при --eval_only → маленькая заглушка
    tr = load_from_disk(os.path.join(_ROOT, args.data_base, "k16", "datasets", "stage2_train"))
    for name in ("orig300", "drop300"):
        tr.select(range(16), new_fingerprint=f"cf_tr_{name}").save_to_disk(
            os.path.join(OUT_BASE, "datasets", name, "stage2_train"))
    nd = [len(json.loads(x["dropped"])) for x in rows_d]
    print(f"   удалено соседей на запрос: {np.mean(nd):.1f}")


def _load_model(args):
    import torch
    from graphvis_lp.config import TrainConfig
    from graphvis_lp.training.families import load_model_and_processor
    from graphvis_lp.training.model import quant_config, vision_precision
    cfg = TrainConfig()
    cfg.model_name = args.model
    cfg.model_family = "auto"
    cfg.vision_bf16 = True
    load_kwargs = {"device_map": "cuda:0"}
    qc = quant_config(cfg)
    if qc is not None:
        load_kwargs["quantization_config"] = qc
    model, processor = load_model_and_processor(cfg, load_kwargs, torch.bfloat16)
    n4, nv = vision_precision(model)
    if n4:
        raise SystemExit(f"vision_bf16 не сработал: {n4}/{nv} Linear зрительной башни в 4-bit")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[{args.label}] adapter: {args.adapter}", flush=True)
    model.eval()
    return model, processor


def cmd_run(args):
    import torch
    from PIL import Image
    out = args.out
    res = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {"scores": {}, "summary": {}}
    if args.label in res["summary"]:
        print(f"[{args.label}] уже посчитано — пропуск")
        print_table(res)
        return
    items = json.load(open(os.path.join(OUT_BASE, "probe_items.json"), encoding="utf-8"))
    for it in items:
        for v, p in list(it["images"].items()) + [("_src", it["orig_image"])]:
            if not os.path.exists(os.path.join(_ROOT, p)):
                raise SystemExit(f"нет картинки {p} — сначала build")
    model, processor = _load_model(args)
    tok = processor.tokenizer
    tok.padding_side = "left"
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")})

    def forward_last(enc):
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                return model(**enc, **{kw: 1}).logits[:, -1]
            except TypeError:
                continue
        return model(**enc).logits[:, -1]

    def score(img, texts):
        vals = []
        for b in range(0, len(texts), args.batch):
            chunk = texts[b:b + args.batch]
            tt = [processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
                tokenize=False, add_generation_prompt=True) for t in chunk]
            enc = processor(text=tt, images=[img] * len(chunk), padding=True,
                            return_tensors="pt").to(model.device)
            with torch.no_grad():
                lg = forward_last(enc).float()
            d = torch.logsumexp(lg[:, yes], -1) - torch.logsumexp(lg[:, no], -1)
            vals += d.cpu().tolist()
        return vals

    t0 = time.time()
    rows = []   # (img_id, task, label, variant_edit, score_edit, score_orig, score_src)
    for n, it in enumerate(items, 1):
        img = {v: Image.open(os.path.join(_ROOT, p)).convert("RGB") for v, p in it["images"].items()}
        src = Image.open(os.path.join(_ROOT, it["orig_image"])).convert("RGB")
        qs = it["questions"]
        texts = [q["text"] for q in qs]
        s_orig = score(img["orig"], texts)
        s_src = score(src, texts)
        for task in TASKS + ("detach_box",):
            js = [j for j, q in enumerate(qs) if q["task"] == task]
            if not js:
                continue
            v = "detach" if task == "detach_box" else task
            s_edit = score(img[v], [texts[j] for j in js])
            for j, se in zip(js, s_edit):
                rows.append([n - 1, task, qs[j]["label"], round(se, 3),
                             round(s_orig[j], 3), round(s_src[j], 3)])
        if n % 20 == 0:
            print(f"  {n}/{len(items)} картинок, {time.time() - t0:.0f}s", flush=True)
    res["scores"][args.label] = rows
    res["summary"][args.label] = summarize(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"saved: {out}  ({time.time() - t0:.0f}s)")
    print_table(res)


def summarize(rows, n_boot=1000, seed=0):
    a = np.array([r[0] for r in rows])
    task = np.array([r[1] for r in rows])
    lab = np.array([r[2] for r in rows])
    se, so, ss = (np.array([r[k] for r in rows], float) for k in (3, 4, 5))
    rng = np.random.default_rng(seed)
    out = {}
    for t in TASKS:
        m = task == t
        if not m.any():
            continue
        ids = np.unique(a[m])
        by = {u: np.where(m & (a == u))[0] for u in ids}
        e, o, s = rcp.auc(se[m], lab[m]), rcp.auc(so[m], lab[m]), rcp.auc(ss[m], lab[m])
        bs = []
        for _ in range(n_boot):
            k = np.concatenate([by[u] for u in rng.choice(ids, len(ids))])
            bs.append(rcp.auc(se[k], lab[k]) - rcp.auc(so[k], lab[k]))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out[t] = {"auc_edit": round(e, 4), "auc_orig": round(o, 4), "auc_src": round(s, 4),
                  "d": round(e - o, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "signif": bool(lo > 0 or hi < 0),
                  "yes_edit_neg_%": round(100 * float(np.mean(se[m & (lab == 0)] > 0)), 1),
                  "yes_orig_neg_%": round(100 * float(np.mean(so[m & (lab == 0)] > 0)), 1),
                  "n_img": int(len(ids)), "n_pos": int(lab[m].sum()), "n_neg": int((1 - lab[m]).sum())}
    m = task == "detach_box"
    if m.any():
        out["detach_box"] = {"yes_edit_%": round(100 * float(np.mean(se[m] > 0)), 1),
                             "yes_orig_%": round(100 * float(np.mean(so[m] > 0)), 1), "n": int(m.sum())}
    return out


def print_table(res):
    print("\n=== CF PROBE: AUC правка / orig (src = исходный PNG), d = правка − orig [95% CI] ===")
    for lbl, s in res["summary"].items():
        for t in TASKS:
            if t not in s:
                continue
            r = s[t]
            print(f"{lbl:14s} {t:7s} AUC {r['auc_edit']:.3f} / {r['auc_orig']:.3f} (src {r['auc_src']:.3f})"
                  f"  d={r['d']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]{' *' if r['signif'] else ''}"
                  f"  yes% на негативах: правка {r['yes_edit_neg_%']} / orig {r['yes_orig_neg_%']}"
                  f"  n={r['n_pos']}+{r['n_neg']} ({r['n_img']} img)")
        if "detach_box" in s:
            b = s["detach_box"]
            print(f"{lbl:14s} box?    отцепленный бокс «есть»: {b['yes_edit_%']}% (orig {b['yes_orig_%']}%)  n={b['n']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=150, help="картинок в пробе")
    b.add_argument("--n_lp", type=int, default=300, help="запросов в LP-подвыборке")
    b.add_argument("--drop_frac", type=float, default=0.3)
    b.add_argument("--size", type=int, default=560)
    b.add_argument("--data_base", default="graphvis_scaling")
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--adapter", default="")
    r.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--out", default=os.path.join("graphvis_readcheck", "cf_probe.json"))
    p = sub.add_parser("report")
    p.add_argument("--out", default=os.path.join("graphvis_readcheck", "cf_probe.json"))
    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        print_table(json.load(open(args.out, encoding="utf-8")))


if __name__ == "__main__":
    main()
