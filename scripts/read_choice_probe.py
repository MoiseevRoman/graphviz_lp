"""Зонд чтения БЕЗ генерации: вопросы Да/Нет, скоринг logit(Yes) − logit(No).

star_read_probe просит модель ПЕРЕЧИСЛИТЬ узлы/тройки — после LP-обучения модель
отвечает одной строкой (format collapse), и зонд мерит формат, а не зрение.
Здесь формат ответа не важен: на каждый вопрос берётся только разница логитов
первого токена ответа, качество — AUC (не зависит от порога и смещения к «Yes»).

Задачи на картинку (те же PNG и seed-42 запросы, что в rank-eval, K=16):
    node      «Есть ли на картинке бокс 'X'?»  позитивы — нарисованные соседи,
              негативы — имена с ДРУГИХ тестовых картинок (тот же стиль/обрезка);
    edge_rel  «Есть ли стрелка 'r' между головой и 'e'?»  позитивы — нарисованные
              пары, негативы — тот же 'e' с ДРУГОЙ нарисованной подписью ребра
              (привязка подписи к боксу);
    edge_ent  те же позитивы против (r, сущность, которой на картинке нет).

Режимы картинки (вопросы одинаковые → сравнение парное):
    real     настоящий рендер;
    shuffle  рендер СЛЕДУЮЩЕГО запроса (как shuffle в rank-eval) — главный контроль:
             модель, выученная на LP, знает соседей головы из KG и без картинки;
    blank    белый холст того же размера.
Чтение = AUC(real) − AUC(shuffle), 95% CI бутстрепом по картинкам.

    python scripts/read_choice_probe.py --adapter graphvis_ckpts_s1/k16/ctx_visual_s1fv/adapter \
        --label s1fv_lp --out graphvis_ckpts_s1/k16/read_choice_probe.json
    python scripts/read_choice_probe.py --dry_run      # только вопросы, без модели
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import star_read_probe as srp   # noqa: E402  (пути/пакет graphvis_lp настраивает он)

Q_NODE = ("The image shows a knowledge graph centered on '{head}'. "
          "Is there a box labeled '{ent}' in the image? Answer Yes or No.")
Q_EDGE = ("The image shows a knowledge graph centered on '{head}'. "
          "Is there an arrow labeled '{rel}' connecting '{head}' and '{ent}'? Answer Yes or No.")
MODES = ("real", "shuffle", "blank")
TASKS = ("node", "edge_rel", "edge_ent")


def build_questions(items, per_img: int, seed: int = 0):
    """Список вопросов [{img, task, label, text}] — одинаковый для всех моделей."""
    rng = random.Random(seed)
    pool = sorted({x for it in items for x in it["names"]})
    qs = []
    for i, it in enumerate(items):
        nb = items[(i + 1) % len(items)]           # shuffle-партнёр
        forbid = {srp._norm(x) for x in it["names"] + nb["names"] + [it["head"], nb["head"]]}
        cand_neg = [x for x in pool if srp._norm(x) not in forbid]
        names = list(dict.fromkeys(it["names"]))
        pos = rng.sample(names, min(per_img, len(names)))
        neg = rng.sample(cand_neg, len(pos))
        h = it["head"]
        qs += [{"img": i, "task": "node", "label": 1, "text": Q_NODE.format(head=h, ent=x)} for x in pos]
        qs += [{"img": i, "task": "node", "label": 0, "text": Q_NODE.format(head=h, ent=x)} for x in neg]

        pairs = list(dict.fromkeys(it["pairs"]))
        pset = {(srp._norm(r), srp._norm(e)) for r, e in pairs}
        rels = sorted({r for r, _ in pairs})
        for r, e in rng.sample(pairs, min(per_img, len(pairs))):
            wrong = [r2 for r2 in rels if (srp._norm(r2), srp._norm(e)) not in pset]
            if not wrong:      # на картинке одна подпись ребра — нечего путать
                continue
            q_pos = Q_EDGE.format(head=h, rel=r, ent=e)
            qs.append({"img": i, "task": "edge_rel", "label": 1, "text": q_pos})
            qs.append({"img": i, "task": "edge_rel", "label": 0,
                       "text": Q_EDGE.format(head=h, rel=rng.choice(wrong), ent=e)})
            qs.append({"img": i, "task": "edge_ent", "label": 1, "text": q_pos})
            qs.append({"img": i, "task": "edge_ent", "label": 0,
                       "text": Q_EDGE.format(head=h, rel=r, ent=rng.choice(cand_neg))})
    return qs


def auc(scores, labels) -> float:
    """Mann–Whitney AUC с учётом ничьих."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if not npos or not nneg:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def summarize(qs, scores, n_boot: int = 1000, seed: int = 0) -> dict:
    """scores[mode] — массив по qs. AUC/acc по задачам и режимам + Δ real−shuffle/blank."""
    rng = np.random.default_rng(seed)
    imgs = np.array([q["img"] for q in qs])
    uimg = np.unique(imgs)
    by_img = {u: np.where(imgs == u)[0] for u in uimg}
    boots = [np.concatenate([by_img[u] for u in rng.choice(uimg, len(uimg))]) for _ in range(n_boot)]
    out = {}
    for t in TASKS:
        sel = np.array([q["task"] == t for q in qs])
        lab = np.array([q["label"] for q in qs])
        res = {}
        for m in MODES:
            s = scores[m]
            res[m] = {"auc": round(auc(s[sel], lab[sel]), 4),
                      "acc_%": round(100 * float(np.mean((s[sel] > 0) == (lab[sel] == 1))), 1),
                      "yes_rate_%": round(100 * float(np.mean(s[sel] > 0)), 1),
                      "n_pos": int(lab[sel].sum()), "n_neg": int((1 - lab[sel]).sum())}
        for base in ("shuffle", "blank"):
            ds = []
            for b in boots:
                bs = b[sel[b]]
                ds.append(auc(scores["real"][bs], lab[bs]) - auc(scores[base][bs], lab[bs]))
            d = res["real"]["auc"] - res[base]["auc"]
            lo, hi = np.nanpercentile(ds, [2.5, 97.5])
            res[f"real-{base}"] = {"d_auc": round(d, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                   "signif": bool(lo > 0 or hi < 0)}
        out[t] = res
    return out


def run(args) -> None:
    import torch
    from PIL import Image
    from graphvis_lp.config import TrainConfig
    from graphvis_lp.training.families import load_model_and_processor
    from graphvis_lp.training.model import quant_config, vision_precision

    tag = args.label or ("base" if not args.adapter else os.path.basename(args.adapter.rstrip("/\\")))
    key = f"{tag}/K{args.k}"
    results = {"summary": {}, "scores": {}}
    if os.path.exists(args.out):
        results = json.load(open(args.out, encoding="utf-8"))
    if key in results["summary"]:
        print(f"[{key}] уже посчитано — пропуск", flush=True)
        print_table(results)
        return

    items = srp.build_items(args.k, args.n, args.data_base, args.max_nodes)
    qs = build_questions(items, args.per_img)
    counts = {t: sum(q["task"] == t for q in qs) for t in TASKS}
    print(f"K={args.k}: {len(items)} images, {len(qs)} questions/mode {counts}", flush=True)

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
        print(f"[{tag}] adapter: {args.adapter}", flush=True)
    model.eval()
    tok = processor.tokenizer
    tok.padding_side = "left"      # последний токен у всех строк батча — позиция ответа
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")})

    def forward_last(enc):
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                return model(**enc, **{kw: 1}).logits[:, -1]
            except TypeError:
                continue
        return model(**enc).logits[:, -1]

    imgs_real = [Image.open(it["image"]).convert("RGB") for it in items]
    blank = Image.new("RGB", imgs_real[0].size, (255, 255, 255))
    by_img = {}
    for j, q in enumerate(qs):
        by_img.setdefault(q["img"], []).append(j)

    scores = {}
    for m in MODES:
        t0 = time.time()
        s = np.zeros(len(qs), dtype=np.float32)
        for done, (i, js) in enumerate(by_img.items(), 1):
            img = {"real": imgs_real[i], "shuffle": imgs_real[(i + 1) % len(items)], "blank": blank}[m]
            uniq = list(dict.fromkeys(qs[j]["text"] for j in js))   # позитивы edge_rel = edge_ent
            val = {}
            for b in range(0, len(uniq), args.batch):
                chunk = uniq[b:b + args.batch]
                texts = [processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}],
                    tokenize=False, add_generation_prompt=True) for t in chunk]
                enc = processor(text=texts, images=[img] * len(chunk), padding=True,
                                return_tensors="pt").to(model.device)
                with torch.no_grad():
                    lg = forward_last(enc).float()
                d = torch.logsumexp(lg[:, yes], -1) - torch.logsumexp(lg[:, no], -1)
                val.update(zip(chunk, d.cpu().tolist()))
            for j in js:
                s[j] = val[qs[j]["text"]]
            if done % 25 == 0:
                print(f"  [{m}] {done}/{len(by_img)} images, {time.time() - t0:.0f}s", flush=True)
        scores[m] = s
        print(f"[{key}] mode={m} done in {time.time() - t0:.0f}s", flush=True)

    summ = summarize(qs, scores)
    results["summary"][key] = summ
    results["scores"][key] = {"questions": [{k: q[k] for k in ("img", "task", "label")} for q in qs],
                              **{m: [round(float(x), 3) for x in scores[m]] for m in MODES}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"saved: {args.out}", flush=True)
    print_table(results)


def print_table(results) -> None:
    print("\n=== READ CHOICE PROBE: AUC (real / shuffle / blank), d = real-shuffle [95% CI] ===")
    for key, summ in results["summary"].items():
        for t in TASKS:
            r = summ[t]
            d = r["real-shuffle"]
            print(f"{key:18s} {t:9s} AUC {r['real']['auc']:.3f} / {r['shuffle']['auc']:.3f} / "
                  f"{r['blank']['auc']:.3f}   d={d['d_auc']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
                  f"{' *' if d['signif'] else ''}   yes%={r['real']['yes_rate_%']}  n={r['real']['n_pos']}+{r['real']['n_neg']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n", type=int, default=100, help="картинок")
    ap.add_argument("--per_img", type=int, default=5, help="позитивов на задачу на картинку")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--adapter", default="", help="PEFT-адаптер (пусто = базовая модель)")
    ap.add_argument("--label", default="")
    ap.add_argument("--data_base", default="graphvis_scaling")
    ap.add_argument("--max_nodes", type=int, default=0,
                    help="как --max_nodes в rerender_dataset (0 = дефолт рендера, 14 узлов)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default="graphvis_ckpts_s1/k16/read_choice_probe.json")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        items = srp.build_items(args.k, args.n, args.data_base, args.max_nodes)
        qs = build_questions(items, args.per_img)
        print(f"{len(items)} images, {len(qs)} questions")
        for t in TASKS:
            sel = [q for q in qs if q["task"] == t]
            print(f"  {t}: {len(sel)} (pos {sum(q['label'] for q in sel)})")
            for q in sel[:4]:
                print(f"    [{q['label']}] {q['text']}")
        return
    run(args)


if __name__ == "__main__":
    main()
