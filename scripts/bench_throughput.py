"""Микро-бенчмарк скорости обучения на ВАШЕЙ карте.

Гоняет ~30 реальных шагов QLoRA на stage2_train, замеряет tokens/s и
экстраполирует на полный прогон (Stage1 1 эпоха + Stage2 2 эпохи).

Usage:
    python -m graphvis_lp.scripts.bench_throughput
"""
from __future__ import annotations
import time, logging, torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

from graphvis_lp.config import TrainConfig
from graphvis_lp.training.dataset import GraphVisLVLMDataset, DataCollatorForGraphVisLVLM
from graphvis_lp.training.families import detect_family
from graphvis_lp.training.model import build_model_and_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Точные размеры собранного датасета (для экстраполяции)
N_STAGE1, N_STAGE2, EP1, EP2 = 155872, 39005, 1, 2
TEXT_TOK = 11_568_608 * EP1 + 4_480_085 * EP2   # текстовые токены с эпохами


def main() -> None:
    cfg = TrainConfig()
    family = detect_family(cfg.model_name, cfg.model_family)
    model, processor, tokenizer = build_model_and_processor(cfg)
    ds = GraphVisLVLMDataset(load_from_disk(f"{cfg.dataset_root}/stage2_train"), processor, tokenizer, cfg)
    collator = DataCollatorForGraphVisLVLM(tokenizer, pad_to_multiple_of=8, family=family)
    loader = DataLoader(ds, batch_size=cfg.per_device_train_batch_size,
                        shuffle=True, collate_fn=collator, num_workers=4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    dev = next(model.parameters()).device

    warmup, measure = 3, 30
    tok_seen, t0 = 0, None
    for i, batch in enumerate(loader):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.time(); tok_seen = 0
        batch = {k: v.to(dev) for k, v in batch.items()}
        opt.zero_grad()
        out = model(**batch); out.loss.backward(); opt.step()
        if i >= warmup:
            tok_seen += int(batch["attention_mask"].sum().item())
        if i == warmup + measure:
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    tps = tok_seen / dt
    print(f"\n=== ЗАМЕР: {tps:,.0f} tokens/s (реально, на этой карте) ===")

    # Фактическое число визуальных токенов на пример из ИЗМЕРЕННОГО батча
    # (для Qwen — динамическое, зависит от min/max_pixels).
    print(f"    (семейство: {family})")

    # Экстраполяция на полный прогон при разном числе визуальных токенов.
    n_pass = N_STAGE1 * EP1 + N_STAGE2 * EP2
    scenarios = [
        ("LLaVA-1.5 (576)", 576),
        ("LLaVA-1.6 (2880)", 2880),
        ("Qwen2.5-VL ~576", 576),
        ("Qwen2.5-VL ~1280", 1280),
    ]
    for name, img in scenarios:
        total = n_pass * img + TEXT_TOK
        hrs = total / tps / 3600
        print(f"  {name}: всего {total/1e6:.0f}M токенов → ~{hrs:.1f} ч "
              f"(если tokens/s держится; при большей длине обычно ниже)")


if __name__ == "__main__":
    main()
