"""Preflight go/no-go перед арендой A100 — прогоняется на локальной GPU (напр. RTX 5070).

Цель: НЕ получить финальные числа, а за ~20-40 мин отсеять неверные развилки и
поймать баги ДО большого запуска. Пять секций, по возрастанию «тяжести»:

  0. ENV      — окружение: CUDA/Blackwell(sm_120), bitsandbytes(4-bit), transformers,
                класс Qwen2.5-VL, flash-attn. Жёсткий gate: без CUDA дальше нет смысла.
  1. ZEROSHOT — базовая (необученная) Qwen на реальных картинках: пол качества +
                сквозная проверка пайплайна (формат промпта, загрузка картинки, парсинг).
  2. INFLUENCE— доходит ли картинка до LM (проба B: ΔNLL/symKL при подмене картинки;
                проба C: доля внимания ответных позиций на визуальные токены, eager).
  3. TRAIN    — микро-обучение QLoRA (batch=1, grad-checkpoint): код-путь без OOM +
                падает ли loss (learning works) + сохраняется/грузится ли LoRA-адаптер.

Каждая секция изолирована try/except и своей загрузкой модели (между секциями
память освобождается), поэтому одна ошибка не роняет остальные. В конце — таблица
PASS/WARN/FAIL и вердикт go/no-go.

Usage (на машине с GPU):
    python scripts/preflight.py                       # дефолты под 12 ГБ
    python scripts/preflight.py --zeroshot_n 100 --probe_n 40 --train_n 200
    python scripts/preflight.py --skip train          # только 0-2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import types
from pathlib import Path

# ── делаем репозиторий импортируемым как пакет `graphvis_lp`, ────────
#    независимо от того, как называется папка (graphviz_lp/graphvis_lp/…).
_REPO = Path(__file__).resolve().parent.parent
if "graphvis_lp" not in sys.modules:
    try:
        if str(_REPO.parent) not in sys.path:
            sys.path.insert(0, str(_REPO.parent))
        __import__("graphvis_lp")
    except Exception:
        _pkg = types.ModuleType("graphvis_lp")
        _pkg.__path__ = [str(_REPO)]          # graphvis_lp.* → файлы репозитория
        sys.modules["graphvis_lp"] = _pkg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preflight")

RESULTS: dict[str, dict] = {}   # section -> {"status": PASS/WARN/FAIL, "note": str, ...}


def _mark(section: str, status: str, note: str, **extra) -> None:
    RESULTS[section] = {"status": status, "note": note, **extra}
    logger.info("[%s] %s — %s", status, section, note)


def _free(*objs) -> None:
    import torch
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────────────
# Ремаппинг путей картинок: в датасете они абсолютные (/workspace/…),
# на чужой машине их надо перебазировать на фактическое расположение репо.
# ────────────────────────────────────────────────────────────────────
def _rebase_images(ds, repo: Path):
    import os
    first = ds[0]["image_path"]
    if os.path.exists(first):
        return ds                      # пути валидны как есть
    marker = "graphvis_fb15k_data"

    def fix(row):
        p = row["image_path"].replace("\\", "/")
        i = p.find(marker)
        if i != -1:
            row["image_path"] = str(repo / p[i:])
        return row

    # new_fingerprint + keep_in_memory: обходим dill-хеширование функции, из-за
    # которого под Python 3.13 всплывает '_thread.RLock … _recursive_count'.
    remapped = ds.map(fix, desc="rebase image paths",
                      keep_in_memory=True, load_from_cache_file=False,
                      new_fingerprint="preflight_rebase")
    ok = os.path.exists(remapped[0]["image_path"])
    logger.info("Перебазировал пути картинок на %s (первый существует: %s)", repo, ok)
    return remapped


def _load_split(cfg, name: str, repo: Path, limit: int | None = None):
    import os
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(cfg.dataset_root, name))
    if limit and limit < len(ds):
        ds = ds.select(range(limit))
    return _rebase_images(ds, repo)


# ════════════════════════════════════════════════════════════════════
# СЕКЦИЯ 0 — окружение
# ════════════════════════════════════════════════════════════════════
def section_env() -> bool:
    print("\n" + "=" * 70 + "\n0. ENV — окружение\n" + "=" * 70)
    ok = True
    info: dict = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if not torch.cuda.is_available():
            _mark("ENV", "FAIL", "CUDA недоступна — GPU-секции запускать бессмысленно", **info)
            return False
        props = torch.cuda.get_device_properties(0)
        cc = f"{props.major}.{props.minor}"
        info.update(gpu=props.name, vram_gb=round(props.total_memory / 1e9, 1),
                    compute_capability=cc, torch_cuda=torch.version.cuda)
        print(f"  GPU        : {props.name}  ({info['vram_gb']} GB, sm_{props.major}{props.minor})")
        print(f"  torch      : {torch.__version__}  (CUDA {torch.version.cuda})")
        # Blackwell = sm_120: нужен torch, собранный с поддержкой этой арки.
        if props.major >= 12:
            archs = torch.cuda.get_arch_list()
            has = any("120" in a or "sm_120" in a for a in archs)
            print(f"  arch_list  : {archs}")
            if not has:
                _mark("ENV", "WARN",
                      f"Blackwell sm_{props.major}{props.minor}, но torch собран без sm_120 "
                      f"({archs}) — вероятны сбои ядра. Нужен torch с CUDA 12.8+.", **info)
                ok = False
        if info["vram_gb"] < 11:
            print(f"  ⚠ VRAM {info['vram_gb']} GB — держите batch=1, grad-checkpoint, умеренный max_pixels.")
    except Exception as e:
        _mark("ENV", "FAIL", f"torch недоступен: {e}")
        return False

    # bitsandbytes (4-bit QLoRA)
    try:
        import bitsandbytes as bnb
        info["bitsandbytes"] = bnb.__version__
        print(f"  bnb        : {bnb.__version__}")
    except Exception as e:
        _mark("ENV", "FAIL", f"bitsandbytes не импортируется (нужен для 4-bit QLoRA): {e}", **info)
        return False

    # transformers + класс Qwen2.5-VL
    try:
        import transformers, peft, accelerate
        info.update(transformers=transformers.__version__, peft=peft.__version__,
                    accelerate=accelerate.__version__)
        print(f"  transformers/{transformers.__version__}  peft/{peft.__version__}  "
              f"accelerate/{accelerate.__version__}")
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        print("  Qwen2_5_VLForConditionalGeneration: доступен")
    except Exception as e:
        _mark("ENV", "FAIL", f"transformers/Qwen2.5-VL класс недоступен: {e}", **info)
        return False

    # flash-attn — опционально; под sm_120 колёс может не быть → sdpa/eager.
    try:
        import flash_attn  # noqa: F401
        info["flash_attn"] = getattr(flash_attn, "__version__", "yes")
        print(f"  flash-attn : {info['flash_attn']}")
    except Exception:
        info["flash_attn"] = None
        print("  flash-attn : нет → используйте attn_implementation='sdpa' (или 'eager' для проб)")

    _mark("ENV", "PASS" if ok else "WARN",
          "окружение готово" if ok else "окружение с оговорками (см. WARN выше)", **info)
    return True


# ════════════════════════════════════════════════════════════════════
# СЕКЦИЯ 1 — zero-shot базлайн + проверка пайплайна
# ════════════════════════════════════════════════════════════════════
def section_zeroshot(cfg, repo: Path, n: int, beams: int):
    print("\n" + "=" * 70 + f"\n1. ZEROSHOT — необученная Qwen, n={n}, beams={beams}\n" + "=" * 70)
    import torch
    from graphvis_lp.training.model import build_model_and_processor
    from graphvis_lp.training.evaluate import evaluate_link_prediction

    model = None
    try:
        cfg.use_lora = False
        cfg.eval_max_samples = n
        cfg.eval_num_beams = beams
        cfg.eval_batch_size = 4
        cfg.eval_max_new_tokens = 16
        test = _load_split(cfg, "stage2_test", repo, limit=max(n * 2, n + 5))
        model, processor, tokenizer = build_model_and_processor(cfg)
        model.eval()
        m = evaluate_link_prediction(model, processor, tokenizer, test, cfg, image_mode="real")
        note = (f"MRR={m['MRR']:.3f} H@1={m['Hits@1']:.1f} H@10={m['Hits@10']:.1f} "
                f"Found={m['Found_Rate']:.1f}%")
        # Пайплайн жив, если генерация вообще выдаёт непустые ответы (Found_Rate замеряет это).
        if m["Found_Rate"] == 0.0:
            _mark("ZEROSHOT", "WARN",
                  "пайплайн отработал, но модель ни разу не угадала (для zero-shot часто норм; "
                  "проверьте, что парсинг ответа осмыслен). " + note, metrics=m)
        else:
            _mark("ZEROSHOT", "PASS", "пайплайн сквозной + ненулевой пол качества. " + note, metrics=m)
    except Exception as e:
        import traceback; traceback.print_exc()
        _mark("ZEROSHOT", "FAIL", f"{type(e).__name__}: {e}")
    finally:
        _free(model)


# ════════════════════════════════════════════════════════════════════
# СЕКЦИЯ 2 — доходит ли картинка до LM (пробы B и C)
# ════════════════════════════════════════════════════════════════════
def section_influence(cfg, repo: Path, n: int):
    print("\n" + "=" * 70 + f"\n2. INFLUENCE — доходит ли картинка до LM, n={n}\n" + "=" * 70)
    import numpy as np
    from graphvis_lp.training.model import build_model_and_processor
    from graphvis_lp.training.families import detect_family
    # image_influence.py лежит рядом; scripts/ не пакет → импортируем по пути.
    if str(_REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(_REPO / "scripts"))
    from image_influence import probe_causal, probe_attention, MODES

    model = None
    try:
        cfg.use_lora = False
        cfg.attn_implementation = "eager"        # обязательно для пробы C (attentions)
        ds = _load_split(cfg, "stage2_test", repo, limit=n * 3)
        family = detect_family(cfg.model_name, cfg.model_family)
        model, processor, tokenizer = build_model_and_processor(cfg)
        model.eval()
        device = next(model.parameters()).device
        idx = sorted(np.random.RandomState(42).choice(len(ds), size=min(n, len(ds)),
                                                       replace=False).tolist())

        b = probe_causal(model, processor, tokenizer, cfg, ds, idx, family, device)
        c = probe_attention(model, processor, tokenizer, cfg, ds, idx, family, device)

        print(f"\n  проба B (ΔNLL при подмене картинки, >0 ⇒ картинка важна):")
        for mode in MODES:
            r = b[mode]
            print(f"    {mode:<8} ΔNLL={r['dNLL_mean']:+.4f}±{r['dNLL_sem']:.4f} "
                  f"symKL={r['symKL_mean']:.4f} helped={100*r['frac_helped']:.0f}%")
        signal = any(b[m]["dNLL_mean"] > 0 and b[m]["dNLL_mean"] > 2 * b[m]["dNLL_sem"]
                     for m in ("blank", "noise"))

        attn_note = ""
        if "error" not in c:
            print(f"  проба C: внимание на картинку ×{c['ratio_over_chance']:.2f} к случайному "
                  f"(доля {c['attn_image_frac_mean']:.3f}, случайно {c['chance_frac_mean']:.3f})")
            attn_note = f" | attn ×{c['ratio_over_chance']:.2f}"
        else:
            print(f"  проба C недоступна: {c['error']}")

        if signal:
            _mark("INFLUENCE", "PASS",
                  "картинка достоверно меняет выход LM (ΔNLL>0 значимо)" + attn_note,
                  probe_B=b, probe_C=c)
        else:
            _mark("INFLUENCE", "WARN",
                  "подмена картинки почти не меняет выход — модель может её игнорировать "
                  "(до обучения это ожидаемо; критично проверить ПОСЛЕ обучения)" + attn_note,
                  probe_B=b, probe_C=c)
    except Exception as e:
        import traceback; traceback.print_exc()
        _mark("INFLUENCE", "FAIL", f"{type(e).__name__}: {e}")
    finally:
        _free(model)


# ════════════════════════════════════════════════════════════════════
# СЕКЦИЯ 3 — микро-обучение QLoRA (smoke + overfit + save/load)
# ════════════════════════════════════════════════════════════════════
def section_train(cfg, repo: Path, n: int, epochs: int):
    print("\n" + "=" * 70 + f"\n3. TRAIN — микро-QLoRA, n={n}, epochs={epochs}\n" + "=" * 70)
    import os, torch
    from graphvis_lp.training.model import build_model_and_processor
    from graphvis_lp.training.dataset import GraphVisLVLMDataset
    from graphvis_lp.training.trainer import make_trainer

    model = None
    adapter_dir = os.path.join(cfg.output_dir, "preflight_adapter")
    try:
        # Конфиг под 12 ГБ: batch=1, accum=4, grad-checkpoint, частые лог/eval.
        cfg.use_lora = True
        cfg.use_4bit = True
        cfg.attn_implementation = None            # для обучения sdpa/дефолт достаточно
        cfg.per_device_train_batch_size = 1
        cfg.per_device_eval_batch_size = 1
        cfg.gradient_accumulation_steps = 4
        cfg.gradient_checkpointing = True
        cfg.stage2_num_epochs = epochs
        cfg.logging_steps = 5
        cfg.eval_steps = max(10, n // (cfg.gradient_accumulation_steps * 2))
        cfg.save_steps = 10_000                   # не сохраняем промежуточно (мало места/времени)
        cfg.save_total_limit = 1
        # 0 воркеров: не форкаем процессы, которые пиклят процессор/токенайзер
        # (внутри замки) — это второй источник RLock-ошибки под Python 3.13.
        cfg.dataloader_num_workers = 0
        cfg.group_by_length = False

        train = _load_split(cfg, "stage2_train", repo, limit=n)
        val = _load_split(cfg, "stage2_valid", repo, limit=max(20, n // 5))

        model, processor, tokenizer = build_model_and_processor(cfg)
        train_ds = GraphVisLVLMDataset(train, processor, tokenizer, cfg)
        val_ds = GraphVisLVLMDataset(val, processor, tokenizer, cfg)

        trainer = make_trainer(model, train_ds, val_ds, cfg, run_name="preflight", tokenizer=tokenizer)
        out = trainer.train()

        # Кривая loss: первое vs последнее логированное значение обучения.
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        first, last = (losses[0], losses[-1]) if len(losses) >= 2 else (float("nan"), float("nan"))
        mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        dropped = len(losses) >= 2 and last < first

        # Сохранение LoRA-адаптера.
        trainer.save_model(adapter_dir)
        files = set(os.listdir(adapter_dir)) if os.path.isdir(adapter_dir) else set()
        saved = "adapter_config.json" in files and any(f.startswith("adapter_model") for f in files)

        print(f"\n  loss: {first:.4f} → {last:.4f}  ({'падает ✓' if dropped else 'НЕ падает ✗'})")
        print(f"  пик VRAM при batch=1: {mem_gb:.1f} GB")
        print(f"  адаптер сохранён ({', '.join(sorted(files)) or 'пусто'}): {saved}")

        _free(trainer, model, train_ds, val_ds)
        model = None

        # Обратная загрузка адаптера на свежую базу + один forward (валидируем reload).
        reloaded = False
        try:
            from peft import PeftModel
            base, processor2, tok2 = build_model_and_processor(
                _clone_cfg(cfg, use_lora=False))
            pm = PeftModel.from_pretrained(base, adapter_dir)
            pm.eval()
            ex = train[0]
            probe_ds = GraphVisLVLMDataset(train.select([0]), processor2, tok2, cfg)
            feat = probe_ds[0]
            from graphvis_lp.training.families import collate as fcollate, detect_family
            batch = fcollate(detect_family(cfg.model_name, cfg.model_family), tok2, [feat], None)
            dev = next(pm.parameters()).device
            batch = {k: v.to(dev) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                _ = pm(**batch)
            reloaded = True
            print("  reload адаптера + forward: ОК")
            _free(pm, base)
        except Exception as e:
            print(f"  reload адаптера: сбой — {type(e).__name__}: {e}")

        note = (f"loss {first:.3f}→{last:.3f} ({'↓' if dropped else 'плоско'}), "
                f"пик VRAM {mem_gb:.1f}GB, save={saved}, reload={reloaded}")
        if dropped and saved and reloaded:
            _mark("TRAIN", "PASS", "код-путь без OOM, loss падает, адаптер сохраняется/грузится. " + note)
        elif not dropped:
            _mark("TRAIN", "WARN", "обучение отработало, но loss не падает — проверьте lr/данные. " + note)
        else:
            _mark("TRAIN", "WARN", "обучение прошло, но save/reload адаптера неполный. " + note)
    except Exception as e:
        import traceback; traceback.print_exc()
        msg = str(e)
        if "out of memory" in msg.lower() or "CUDA out of memory" in msg:
            _mark("TRAIN", "FAIL", "OOM даже при batch=1 — снизьте qwen_max_pixels/max_length. " + msg[:160])
        else:
            _mark("TRAIN", "FAIL", f"{type(e).__name__}: {msg[:200]}")
    finally:
        _free(model)


def _clone_cfg(cfg, **overrides):
    import dataclasses
    return dataclasses.replace(cfg, **overrides)


# ════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--zeroshot_n", type=int, default=120)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--probe_n", type=int, default=50)
    ap.add_argument("--train_n", type=int, default=200)
    ap.add_argument("--train_epochs", type=int, default=2)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["zeroshot", "influence", "train"],
                    help="какие секции пропустить")
    ap.add_argument("--out", default=str(_REPO / "graphvis_ckpts" / "preflight_report.json"))
    args = ap.parse_args()

    from graphvis_lp.config import TrainConfig

    def fresh_cfg():
        cfg = TrainConfig()
        cfg.dataset_root = str(_REPO / "graphvis_fb15k_data" / "datasets")
        cfg.image_root = ""
        cfg.output_dir = str(_REPO / "graphvis_ckpts" / "preflight")
        cfg.model_name = args.model
        cfg.model_family = "auto"
        return cfg

    # 0. ENV — жёсткий gate.
    if not section_env():
        _dump(args.out)
        print("\nENV FAIL — GPU-секции пропущены."); return

    # 1-3. Каждая секция со своим свежим cfg (изоляция от мутаций).
    if "zeroshot" not in args.skip:
        section_zeroshot(fresh_cfg(), _REPO, args.zeroshot_n, args.beams)
    if "influence" not in args.skip:
        section_influence(fresh_cfg(), _REPO, args.probe_n)
    if "train" not in args.skip:
        section_train(fresh_cfg(), _REPO, args.train_n, args.train_epochs)

    _dump(args.out)
    _verdict()


def _dump(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    logger.info("Отчёт сохранён: %s", path)


def _verdict():
    print("\n" + "=" * 70 + "\nИТОГ preflight (go/no-go перед A100)\n" + "=" * 70)
    order = ["ENV", "ZEROSHOT", "INFLUENCE", "TRAIN"]
    icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}
    for s in order:
        if s in RESULTS:
            r = RESULTS[s]
            print(f"  {icon.get(r['status'], '  ')} {s:<10} {r['note']}")
    statuses = [RESULTS[s]["status"] for s in order if s in RESULTS]
    if "FAIL" in statuses:
        print("\n❌ NO-GO: устраните FAIL перед арендой A100.")
    elif "WARN" in statuses:
        print("\n⚠️  GO с оговорками: разберите WARN (часть — ожидаемы для zero-shot).")
    else:
        print("\n✅ GO: пайплайн готов к полному запуску на A100.")
    print("=" * 70)


if __name__ == "__main__":
    main()
