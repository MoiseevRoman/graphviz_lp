"""Гейт жизнеспособности: даёт ли ДООБУЧЕНИЕ сигнал вообще — прогон на 5070 до A100.

Все зонды (preflight, legibility, neighbor_ranking) меряли НЕОБУЧЕННУЮ модель и
принципиально не отвечают на вопрос «заработает ли метод после дообучения».
Отвечает только реальное дообучение. Этот скрипт:

  1. дообучает QLoRA на N_train примерах (1 эпоха, конфиг под 12 ГБ);
  2. генеративно оценивает MRR/Hits@1/3/10 на N_test тест-примерах;
  3. сравнивает с zero-shot полом (≈0 из preflight).

Интерпретация:
  • качество ЗАМЕТНО выше нуля → премиса подтверждена, масштабирование на 40k/A100
    оправдано; можно переходить к перерендеру в целевом конфиге и полному запуску.
  • качество near-zero → это НЕ окончательный приговор (здесь используются УЖЕ
    отрендеренные картинки старого конфига semantic@12/336px — худший вариант),
    но повод сперва перерендерить в hybrid@4/896 и повторить, а НЕ жечь A100.

По умолчанию берёт существующий датасет (semantic@12, 336px) — это самый быстрый
сигнал без перерендера. Положительный результат на ХУДШЕМ конфиге особенно убедителен.

Usage:
    python scripts/viability_gate.py                       # 4000 train / 400 test
    python scripts/viability_gate.py --train_n 6000 --eval_n 500 --epochs 1
"""

from __future__ import annotations

# Windows-консоль и редирект в файл по умолчанию cp1251 → печать символа '→' и
# части юникода роняла процесс UnicodeEncodeError (краш ПОСЛЕ сохранения отчёта).
# Переводим stdout/stderr на utf-8 с errors=replace, чтобы печать/логи никогда не
# валили процесс и русский текст в логах читался.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import gc
import json
import logging
import os
import sys
import types
from pathlib import Path

# Меньше фрагментации VRAM на 12 ГБ (важно для eval сразу после обучения).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── делаем репозиторий импортируемым как `graphvis_lp` независимо от имени папки ──
_REPO = Path(__file__).resolve().parent.parent
if "graphvis_lp" not in sys.modules:
    try:
        if str(_REPO.parent) not in sys.path:
            sys.path.insert(0, str(_REPO.parent))
        __import__("graphvis_lp")
    except Exception:
        _pkg = types.ModuleType("graphvis_lp")
        _pkg.__path__ = [str(_REPO)]
        sys.modules["graphvis_lp"] = _pkg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viability")


def _rebase_images(ds, repo: Path):
    """Пути картинок в датасете абсолютные (/workspace/…); перебазируем на репо."""
    first = ds[0]["image_path"]
    if os.path.exists(first):
        return ds
    marker = "graphvis_fb15k_data"

    def fix(row):
        p = row["image_path"].replace("\\", "/")
        i = p.find(marker)
        if i != -1:
            row["image_path"] = str(repo / p[i:])
        return row

    return ds.map(fix, desc="rebase image paths", keep_in_memory=True,
                  load_from_cache_file=False, new_fingerprint="viability_rebase")


def _load_split(cfg, name, repo, limit=None):
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(cfg.dataset_root, name))
    if limit and limit < len(ds):
        ds = ds.select(range(limit))
    return _rebase_images(ds, repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--train_n", type=int, default=4000)
    ap.add_argument("--eval_n", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--beams", type=int, default=10)
    ap.add_argument("--eval_mode", choices=["rank", "generative", "sample"], default="rank",
                    help="rank = filtered likelihood-ranking среди ~50 кандидатов (быстро, "
                         "НЕ сравнимо с SOTA); generative = beam-поиск (медленно); "
                         "sample = KGT5-протокол (открытое сэмплирование, ранг среди всех |E| — "
                         "сравнимо с таблицами KGT5/KGT5-context)")
    # ── параметры sample-режима (KGT5) ──
    ap.add_argument("--num_samples", type=int, default=500,
                    help="сэмплов на запрос (KGT5=500). Меньше → быстрее, ниже recall")
    ap.add_argument("--sample_temperature", type=float, default=1.0,
                    help="температура сэмплирования (>1 = больше разнообразие/coverage)")
    ap.add_argument("--sample_top_p", type=float, default=1.0, help="nucleus top-p (1.0 = выкл)")
    ap.add_argument("--gen_batch", type=int, default=50,
                    help="сколько сэмплов генерировать одним батчем (VRAM: 12ГБ≈25–50, A100≈200)")
    ap.add_argument("--sample_max_new_tokens", type=int, default=16, help="макс. длина имени-хвоста")
    ap.add_argument("--sample_conditions",
                    default="text-only,text-context,image-context,image+text",
                    help="строки «зоны B» через запятую: text-only|text-context|image-context|"
                         "image+text (2×2: картинка real/blank × промпт minimal/full)")
    ap.add_argument("--cand_cap", type=int, default=50,
                    help="сколько type-constrained кандидатов-хвостов на запрос (rank)")
    ap.add_argument("--cand_batch", type=int, default=8, help="батч кандидатов при скоринге (rank)")
    ap.add_argument("--image_modes", default="real",
                    help="режимы картинки через запятую для абляции: real,blank,noise,shuffle "
                         "(разрыв real−blank = вклад изображения)")
    ap.add_argument("--eval_only", action="store_true",
                    help="пропустить обучение, загрузить сохранённый адаптер и только оценить")
    ap.add_argument("--train_vision", action="store_true",
                    help="разморозить зрительную башню (LoRA на ViT+merger) — чтобы картинка МОГЛА работать")
    ap.add_argument("--lora_r", type=int, default=0,
                    help="ранг LoRA (0 = дефолт конфига 64; уменьшить при OOM с train_vision)")
    ap.add_argument("--with_stage1", action="store_true",
                    help="curriculum: сперва обучить на stage1_graph_comprehension "
                         "(ЧТЕНИЕ графа: счёт узлов/рёбер, степень, списки узлов и троек), "
                         "затем stage2 link prediction на ТОЙ ЖЕ модели. stage1 ВСЕГДА на "
                         "реальных картинках. Нужен собранный datasets/stage1_graph_comprehension "
                         "(scripts/build_stage1_from_stage2.py). Игнорируется с --eval_only.")
    ap.add_argument("--stage1_epochs", type=int, default=1, help="эпох на stage1 (по умолчанию 1)")
    ap.add_argument("--stage1_n", type=int, default=0,
                    help="ограничить число stage1-примеров (0 = все ~48k). Примеры сгруппированы "
                         "по графам подряд → первые N ≈ первые N/8 графов (частичное покрытие).")
    ap.add_argument("--minimal_prompt", action="store_true",
                    help="убрать 'Entity description: …' из промптов (train+eval): окрестность только из картинки")
    ap.add_argument("--context_mode",
                    choices=["none", "text", "visual", "both", "text_feat", "both_feat",
                             "text_adjlist", "text_matrix"], default=None,
                    help="как подаётся окрестность (train+eval, пересобирает промпт): "
                         "none=без контекста; text=соседи ТЕКСТОМ+blank-картинка (≈KGT5-context); "
                         "visual=картинка+minimal-промпт (наш метод); both=и то, и другое; "
                         "text_feat=текст соседей + инженерные признаки движка (degree/центральность/"
                         "[layout]) + blank-картинка — проверка гипотезы inductive bias; "
                         "both_feat=картинка + текст-с-признаками. "
                         "text/none/text_feat обучаются с blank-картинкой. Нужна колонка "
                         "neighbors_text (text/both) или neighbors_feat (text_feat/both_feat) — ре-рендер.")
    ap.add_argument("--output_dir", default=str(_REPO / "graphvis_ckpts" / "viability"),
                    help="куда сохранять адаптер/метрики (разные эксперименты → разные папки)")
    ap.add_argument("--data_dir", default=str(_REPO / "FB15k-237"),
                    help="исходные тройки FB15k-237 (для кандидатов и filtered, rank)")
    ap.add_argument("--dataset_root", default=str(_REPO / "graphvis_fb15k_data" / "datasets"),
                    help="датасет для гейта (напр. перерендеренный hybrid@4/672)")
    ap.add_argument("--out", default=str(_REPO / "graphvis_ckpts" / "viability_report.json"))
    ap.add_argument("--adapter_dir", default="",
                    help="грузить адаптер отсюда (eval-only), а метрики писать в --output_dir; "
                         "по умолчанию адаптер берётся из <output_dir>/adapter. "
                         "Позволяет переоценить готовый чекпойнт, не перезаписывая его результаты.")
    args = ap.parse_args()

    import torch
    from graphvis_lp.config import TrainConfig
    from graphvis_lp.training.model import build_model_and_processor
    from graphvis_lp.training.dataset import GraphVisLVLMDataset
    from graphvis_lp.training.trainer import make_trainer
    from graphvis_lp.training.evaluate import (
        evaluate_link_prediction, evaluate_link_prediction_rank,
        evaluate_link_prediction_sample)

    cfg = TrainConfig()
    cfg.dataset_root = args.dataset_root
    # Датасеты хранят ОТНОСИТЕЛЬНЫЙ image_path (от корня репо) → резолвим от _REPO,
    # чтобы работало независимо от CWD и на Windows, и на Linux. Абсолютные пути
    # старых датасетов os.path.join игнорирует (absolute wins) → не ломается.
    cfg.image_root = str(_REPO)
    cfg.output_dir = args.output_dir
    cfg.model_name = args.model
    cfg.model_family = "auto"
    cfg.train_vision = args.train_vision
    if args.lora_r > 0:
        cfg.lora_r = args.lora_r
        cfg.lora_alpha = args.lora_r * 2

    # ── конфиг обучения под 12 ГБ ──
    cfg.use_lora = True
    cfg.use_4bit = True
    cfg.per_device_train_batch_size = 1
    cfg.gradient_accumulation_steps = 8
    cfg.gradient_checkpointing = True
    cfg.stage2_num_epochs = args.epochs
    cfg.logging_steps = 20
    cfg.eval_steps = 10_000          # без eval-во-время-обучения (оцениваем после)
    cfg.save_steps = 10_000
    cfg.save_total_limit = 1
    cfg.dataloader_num_workers = 0   # избегаем RLock-глюка multiprocess под Py3.13
    cfg.group_by_length = False

    if not torch.cuda.is_available():
        logger.error("CUDA недоступна — гейт нужно запускать на машине с GPU.")
        sys.exit(1)

    train = _load_split(cfg, "stage2_train", _REPO, limit=args.train_n)
    test = _load_split(cfg, "stage2_test", _REPO, limit=max(args.eval_n * 2, args.eval_n + 5))

    if args.minimal_prompt and args.eval_mode == "sample":
        logger.warning("--minimal_prompt игнорируется в sample-режиме: он сам строит "
                       "full/minimal-варианты промпта для условий text-context/image-context.")

    if args.minimal_prompt and args.eval_mode != "sample":
        import re as _re
        _desc = _re.compile(r"Entity description:.*?(?=\n\s*Query:)", _re.S)

        def _strip(ds):
            return ds.map(lambda r: {"prompt": _desc.sub("", r["prompt"])},
                          desc="strip description", keep_in_memory=True,
                          load_from_cache_file=False)

        train, test = _strip(train), _strip(test)
        logger.info("minimal_prompt: описание сущности удалено из промптов (train+eval)")
        if args.eval_only:
            logger.warning("--minimal_prompt + --eval_only: адаптер мог быть обучен С описанием "
                           "→ рассогласование train/eval. Обычно minimal_prompt нужен ПРИ обучении.")

    if args.context_mode:
        # Пересобираем промпт под режим подачи окрестности (train+eval идентично).
        import neighbor_ranking_eval as _nre
        mode = args.context_mode
        # text_<enc>: разные текстовые сериализации ОДНОГО подграфа (те же рёбра) —
        # для чистого сравнения представлений графа текстом.
        _TEXT_ENC_COL = {"text_adjlist": "neighbors_adjlist", "text_matrix": "neighbors_matrix"}
        needs_text = mode in ("text", "both", "text_feat", "both_feat",
                              "text_adjlist", "text_matrix")
        use_feat = mode in ("text_feat", "both_feat")
        # Для *_feat берём колонку с признаками; для text_<enc> — колонку кодировки;
        # если её нет — деградируем к neighbors_text (и предупреждаем), чтобы не падать.
        nbr_col = "neighbors_feat" if use_feat else _TEXT_ENC_COL.get(mode, "neighbors_text")
        if needs_text and nbr_col not in train.column_names:
            if (use_feat or mode in _TEXT_ENC_COL) and "neighbors_text" in train.column_names:
                logger.warning("Нет колонки %s — откат к neighbors_text. Для этой кодировки "
                               "запусти scripts/add_text_encodings.py по датасету.", nbr_col)
                nbr_col = "neighbors_text"
            else:
                logger.error("В датасете нет колонки %s — нужен ре-рендер обновлённым "
                             "rerender_dataset.py. Пересобери датасет.", nbr_col)
                sys.exit(1)
        if args.minimal_prompt:
            logger.warning("--context_mode переопределяет промпт целиком; --minimal_prompt игнорируется.")

        def _rebuild(row):
            head = row["head"]
            rel = _nre._lastseg(row["relation"]).replace("_", " ")
            nbr = row.get(nbr_col, "") if needs_text else ""
            q = (f"Query: ({head}, {rel}, ?)\n"
                 f"Predict the missing tail entity. Answer with the entity name only.")
            if mode == "none":
                ctx = ""
            elif mode in ("text", "text_feat", "text_adjlist", "text_matrix"):
                ctx = f"Known facts about {head}:\n{nbr}\n\n" if nbr else ""
            elif mode == "visual":
                ctx = f"The image shows the 1-hop neighborhood of {head} in a knowledge graph.\n\n"
            else:  # both / both_feat
                ctx = (f"The image shows the 1-hop neighborhood of {head}.\n"
                       f"Known facts about {head}:\n{nbr}\n\n" if nbr
                       else f"The image shows the 1-hop neighborhood of {head}.\n\n")
            return {"prompt": ctx + q}

        train = train.map(_rebuild, desc=f"prompt[{mode}]", keep_in_memory=True, load_from_cache_file=False)
        test = test.map(_rebuild, desc=f"prompt[{mode}]", keep_in_memory=True, load_from_cache_file=False)
        cfg.train_image_mode = "blank" if mode in ("none", "text", "text_feat",
                                                   "text_adjlist", "text_matrix") else "real"
        logger.info("context_mode=%s: промпты пересобраны; train_image_mode=%s (соседей текстом: %s)",
                    mode, cfg.train_image_mode, needs_text)

    adapter = args.adapter_dir or os.path.join(cfg.output_dir, "adapter")
    s1_first = s1_last = None   # лоссы stage1 (заполняются только при --with_stage1)
    train_budget = {}           # бюджет обучения в токенах (заполняется при обучении)

    if args.eval_only:
        if not os.path.isdir(adapter):
            logger.error("Нет адаптера в %s — сначала обучение (без --eval_only).", adapter)
            sys.exit(1)
        logger.info("EVAL-ONLY: загружаю сохранённый адаптер %s", adapter)
        cfg.use_lora = False   # база + PEFT-адаптер поверх
        model, processor, tokenizer = build_model_and_processor(cfg)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        first = last = float("nan")
        peak_gb = 0.0
    else:
        model, processor, tokenizer = build_model_and_processor(cfg)

        # ── Stage 1 (опционально): обучение ЧТЕНИЮ графа перед link prediction ──
        # Тот же model-объект → адаптер переносится в stage2 (curriculum, как в
        # train_and_eval). stage1 всегда на РЕАЛЬНЫХ картинках, иначе учить читать
        # нечего; режим картинки для stage2 восстанавливаем после.
        if args.with_stage1:
            s1_dir = os.path.join(cfg.dataset_root, "stage1_graph_comprehension")
            if not os.path.isdir(s1_dir) and not os.path.isdir(str(_REPO / s1_dir)):
                logger.error("Нет stage1_graph_comprehension в %s — собери его "
                             "(scripts/build_stage1_from_stage2.py).", s1_dir)
                sys.exit(1)
            s1_raw = _load_split(cfg, "stage1_graph_comprehension", _REPO,
                                 limit=(args.stage1_n or None))
            # У stage1-промптов префикс "<image>\n" (наследие LLaVA). Для Qwen это
            # лишний текстовый токен — снимаем; families.format_eval_text добавит
            # картинку сам (chat-template у Qwen, "<image>\n" у LLaVA).
            import re as _re_img
            _img_pref = _re_img.compile(r"^\s*<image>\s*")
            s1_raw = s1_raw.map(lambda r: {"prompt": _img_pref.sub("", r["prompt"])},
                                desc="strip <image>", keep_in_memory=True,
                                load_from_cache_file=False)
            cfg.stage1_num_epochs = args.stage1_epochs
            _prev_img_mode = cfg.train_image_mode
            cfg.train_image_mode = "real"
            s1_ds = GraphVisLVLMDataset(s1_raw, processor, tokenizer, cfg)
            logger.info("=" * 64)
            logger.info("STAGE 1 (graph comprehension): %d примеров, %d эпоха(и), image=real",
                        len(s1_ds), cfg.stage1_num_epochs)
            logger.info("=" * 64)
            tr1 = make_trainer(model, s1_ds, None, cfg, run_name="stage1_viability",
                               tokenizer=tokenizer)
            tr1.train()
            _l1 = [h["loss"] for h in tr1.state.log_history if "loss" in h]
            s1_first, s1_last = (_l1[0], _l1[-1]) if len(_l1) >= 2 else (float("nan"), float("nan"))
            logger.info("STAGE 1 готов: loss %.3f → %.3f", s1_first, s1_last)
            del tr1
            gc.collect()
            torch.cuda.empty_cache()
            cfg.train_image_mode = _prev_img_mode   # вернуть режим картинки для stage2

        train_ds = GraphVisLVLMDataset(train, processor, tokenizer, cfg)

        logger.info("=" * 64)
        logger.info("ОБУЧЕНИЕ (stage2): %d примеров, %d эпоха(и), batch=1×accum=%d",
                    len(train_ds), args.epochs, cfg.gradient_accumulation_steps)
        logger.info("=" * 64)
        trainer = make_trainer(model, train_ds, None, cfg, run_name="viability", tokenizer=tokenizer)
        trainer.train()

        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        first, last = (losses[0], losses[-1]) if len(losses) >= 2 else (float("nan"), float("nan"))
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        trainer.save_model(adapter)

        # ── бюджет ОБУЧЕНИЯ в токенах (для честного сравнения «при равных мощностях») ──
        # train_n одинаков across conditions, но токенов на пример у модальностей
        # разное → total training tokens различается. Логируем, чтобы сравнение
        # можно было привести к равному обучающему бюджету (или проверить iso-token).
        try:
            from graphvis_lp.token_cost import measure_token_cost
            _tc = measure_token_cost(processor, tokenizer, train, cfg,
                                     prompt_key="prompt",
                                     image_mode=getattr(cfg, "train_image_mode", "real"))
            _ntr = len(train)
            train_budget = {
                "train_mean_tokens": _tc.get("mean_total_tokens", 0.0),
                "train_mean_text_tokens": _tc.get("mean_text_tokens", 0.0),
                "train_mean_visual_tokens": _tc.get("mean_visual_tokens", 0.0),
                "train_examples": _ntr,
                "train_epochs": args.epochs,
                "train_tokens_total": _tc.get("mean_total_tokens", 0.0) * _ntr * args.epochs,
            }
            logger.info("Бюджет обучения: %.0f токенов/пример × %d × %d эпох = %.2fM токенов",
                        train_budget["train_mean_tokens"], _ntr, args.epochs,
                        train_budget["train_tokens_total"] / 1e6)
        except Exception as exc:
            logger.warning("train token budget measure failed: %s", exc)

        # Освобождаем VRAM обучения (оптимизатор/градиенты/ссылки Trainer) до оценки.
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # ── оценка на тесте ──
    cfg.eval_max_samples = args.eval_n
    model.eval()
    logger.info("=" * 64)
    mode_metrics = {}
    if args.eval_mode == "rank":
        modes = [x.strip() for x in args.image_modes.split(",") if x.strip()]
        if args.context_mode in ("none", "text", "text_feat", "text_adjlist", "text_matrix"):
            modes = ["blank"]   # текстовые условия: картинка всегда пустая (train=eval)
        logger.info("ОЦЕНКА (rank, filtered): %d примеров, cand_cap=%d, режимы=%s",
                    args.eval_n, args.cand_cap, modes)
        logger.info("=" * 64)
        import textlp_upperbound as tub   # переиспользуем карты имён/кандидатов
        rel2tails, hr2tails = tub.build_name_maps(args.data_dir)

        def candidate_provider(sample):
            head = sample["head"]
            rq = tub.nre._lastseg(sample["relation"])
            gold = sample["tail"]
            cands = tub.make_candidates(rq, gold, rel2tails, args.cand_cap)
            if gold not in cands or len(cands) < 2:
                return None
            filt = hr2tails.get((head, rq), set()) - {gold}
            return cands, filt, gold

        for mode in modes:
            logger.info("─── image_mode=%s ───", mode)
            mode_metrics[mode] = evaluate_link_prediction_rank(
                model, processor, tokenizer, test, cfg, candidate_provider,
                image_mode=mode, cand_batch=args.cand_batch)
        m = mode_metrics.get("real", next(iter(mode_metrics.values())))
    elif args.eval_mode == "sample":
        import textlp_upperbound as tub
        rel2tails, hr2tails = tub.build_name_maps(args.data_dir)
        valid_names, n_entities = tub.build_entity_universe(args.data_dir)
        logger.info("ОЦЕНКА (sample/KGT5): %d примеров, N=%d, |E|=%d, условия=%s",
                    args.eval_n, args.num_samples, n_entities, args.sample_conditions)
        logger.info("=" * 64)

        # Минимальный промпт (без вербализованной окрестности) как отдельная колонка,
        # чтобы одним прогоном сравнить контекст-в-тексте vs контекст-в-картинке.
        import re as _re
        _desc = _re.compile(r"Entity description:.*?(?=\n\s*Query:)", _re.S)
        test_s = test.map(lambda r: {"prompt_min": _desc.sub("", r["prompt"])},
                          desc="build prompt_min", keep_in_memory=True,
                          load_from_cache_file=False)

        def filter_provider(sample):
            head = sample["head"]
            rq = tub.nre._lastseg(sample["relation"])
            gold = sample["tail"]
            return gold, hr2tails.get((head, rq), set())

        if args.context_mode:
            # Открытый (KGT5) eval чекпойнта, обученного в конкретном context_mode:
            # одно условие, согласованное с обучением (та же картинка real/blank и
            # тот же пересобранный промпт), метрика прямо сравнима с KGT5/KGT5-context.
            # Встроенный 2×2 ниже рассчитан на «описание-как-контекст» и на
            # neighbors-датасете (описания вырезаны) вырождается — потому обходим его.
            COND = {args.context_mode: (cfg.train_image_mode, "prompt")}
            conds = [args.context_mode]
        else:
            # 2×2: картинка real/blank × промпт full/minimal (описание-как-контекст)
            COND = {
                "text-only":     ("blank", "prompt_min"),   # ≈ KGT5 (нет окрестности)
                "text-context":  ("blank", "prompt"),       # ≈ KGT5-context (окрестность текстом)
                "image-context": ("real",  "prompt_min"),   # окрестность ТОЛЬКО картинкой
                "image+text":    ("real",  "prompt"),       # картинка + текст
            }
            conds = [c.strip() for c in args.sample_conditions.split(",") if c.strip() in COND]
        for c in conds:
            img_mode, pkey = COND[c]
            logger.info("─── условие «%s» (image=%s, prompt=%s) ───", c, img_mode, pkey)
            mode_metrics[c] = evaluate_link_prediction_sample(
                model, processor, tokenizer, test_s, cfg,
                valid_names=valid_names, n_entities=n_entities,
                filter_provider=filter_provider, image_mode=img_mode, prompt_key=pkey,
                num_samples=args.num_samples, temperature=args.sample_temperature,
                top_p=args.sample_top_p, gen_batch=args.gen_batch,
                max_new_tokens=args.sample_max_new_tokens, label=c)
        m = mode_metrics.get("image+text", next(iter(mode_metrics.values())))
    else:
        cfg.eval_num_beams = args.beams
        cfg.eval_batch_size = 4
        cfg.eval_max_new_tokens = 24
        logger.info("ОЦЕНКА (generative): %d тест-примеров, beams=%d", args.eval_n, args.beams)
        logger.info("=" * 64)
        m = evaluate_link_prediction(model, processor, tokenizer, test, cfg, image_mode="real")

    report = {
        "config": {"train_n": (0 if args.eval_only else len(train_ds)),
                   "eval_only": args.eval_only, "adapter": adapter,
                   "eval_n": args.eval_n, "epochs": args.epochs,
                   "eval_mode": args.eval_mode, "beams": args.beams, "cand_cap": args.cand_cap,
                   "dataset_root": args.dataset_root, "model": cfg.model_name},
        "train": {"loss_first": first, "loss_last": last, "peak_vram_gb": round(peak_gb, 1),
                  "with_stage1": args.with_stage1,
                  "stage1_loss_first": s1_first, "stage1_loss_last": s1_last,
                  **train_budget},
        "eval": m,
        "ablation": mode_metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)

    # ── вердикт ──
    if "Found_Rate" in m:
        extra = f"Found={m['Found_Rate']:.1f}%"
    elif args.eval_mode == "sample":
        extra = f"n={m.get('n_evaluated', '?')}  gold-sampled={m.get('sampled_gold_rate', 0):.0f}%"
    else:
        extra = f"n={m.get('n_evaluated', '?')}  med.cand={m.get('mean_candidates', 0):.0f}"
    print("\n" + "=" * 64)
    print("ГЕЙТ ЖИЗНЕСПОСОБНОСТИ  (%s%s)" % (
        args.eval_mode, ", filtered" if args.eval_mode in ("rank", "sample") else ""))
    print("=" * 64)
    if args.with_stage1 and s1_first is not None:
        print(f"  stage1   : loss {s1_first:.3f} → {s1_last:.3f}  (чтение графа)")
    print(f"  обучение : loss {first:.3f} → {last:.3f} | пик VRAM {peak_gb:.1f} ГБ")
    print(f"  качество : MRR={m['MRR']:.4f}  H@1={m['Hits@1']:.1f}  "
          f"H@3={m['Hits@3']:.1f}  H@10={m['Hits@10']:.1f}  {extra}")
    print(f"  zero-shot пол (preflight) : MRR≈0")
    print("-" * 64)
    if m["Hits@10"] >= 5.0 or m["MRR"] >= 0.03:
        print("  ✅ ЕСТЬ СИГНАЛ: дообучение заметно поднимает качество над нулём.")
        print("     → премиса подтверждена, можно масштабировать на полный тест / A100.")
    else:
        print("  ⚠️  СИГНАЛ СЛАБЫЙ на этом конфиге/датасете.")
        print("     → проверить данные (читаемость, ранжирование соседей) до полного запуска.")
    print("=" * 64)

    # ── sample/KGT5: таблица «зоны B» + сравнимость ──
    if args.eval_mode == "sample" and mode_metrics:
        print("\nЗОНА B — единый KGT5-протокол (сравнимо с KGT5/KGT5-context)")
        print(f"{'условие':<15}{'MRR':>9}{'H@1':>8}{'H@3':>8}{'H@10':>8}{'gold%':>8}{'n':>6}")
        print("-" * 62)
        for c, mm in mode_metrics.items():
            print(f"{c:<15}{mm['MRR']:>9.4f}{mm['Hits@1']:>8.1f}{mm['Hits@3']:>8.1f}"
                  f"{mm['Hits@10']:>8.1f}{mm.get('sampled_gold_rate', 0):>8.1f}{mm.get('n_evaluated', 0):>6}")
        print("-" * 62)
        g = mode_metrics.get
        def _d(a, b, key="MRR"):
            if g(a) and g(b):
                return g(a)[key] - g(b)[key]
            return None
        pairs = [("text-context", "text-only", "окрестность текстом даёт"),
                 ("image-context", "text-only", "окрестность картинкой даёт"),
                 ("image-context", "text-context", "картинка ПРОТИВ текста (ядро тезиса)"),
                 ("image+text", "text-context", "картинка СВЕРХ текста даёт")]
        for a, b, msg in pairs:
            d = _d(a, b)
            if d is not None:
                print(f"  {a} − {b}: ΔMRR={d:+.4f}  → {msg}")
        print("(|E|=%d; sampled%% = доля запросов, где золотой вообще попал в сэмплы — "
              "потолок recall при данном N/температуре)" % m.get("n_entities", 0))

    # ── абляция модальности: вклад изображения (real − blank/shuffle) ──
    if args.eval_mode == "rank" and len(mode_metrics) > 1:
        print("\nАБЛЯЦИЯ МОДАЛЬНОСТИ (вклад изображения)")
        print(f"{'режим':<10}{'MRR':>9}{'H@1':>8}{'H@3':>8}{'H@10':>8}{'n':>7}")
        print("-" * 50)
        for mode, mm in mode_metrics.items():
            print(f"{mode:<10}{mm['MRR']:>9.4f}{mm['Hits@1']:>8.1f}"
                  f"{mm['Hits@3']:>8.1f}{mm['Hits@10']:>8.1f}{mm.get('n_evaluated', 0):>7}")
        real = mode_metrics.get("real")
        if real:
            print("-" * 50)
            for base in ("blank", "noise", "shuffle"):
                if base in mode_metrics:
                    d = real["MRR"] - mode_metrics[base]["MRR"]
                    dh = real["Hits@1"] - mode_metrics[base]["Hits@1"]
                    print(f"  real − {base:<8}: ΔMRR={d:+.4f}  ΔHits@1={dh:+.1f}  "
                          f"→ {'изображение ПОМОГАЕТ' if d > 0.01 else 'вклад картинки мал'}")
    print("Сохранено:", args.out)


if __name__ == "__main__":
    main()
