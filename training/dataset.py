from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from graphvis_lp.config import TrainConfig
from graphvis_lp.training.families import (
    collate as family_collate,
    detect_family,
    image_token_str,
    process_train_example,
)

logger = logging.getLogger(__name__)

_FALLBACK_COLOR = (128, 128, 128)


class GraphVisLVLMDataset(TorchDataset):
    """Dataset, который на лету строит ``(input_ids, labels, pixel_values)``."""

    def __init__(self, hf_dataset, processor, tokenizer, cfg: TrainConfig):
        self.ds = hf_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.family = detect_family(cfg.model_name, cfg.model_family)
        self._image_token: str = image_token_str(processor, self.family)
        self._lengths: list | None = None

    def __len__(self) -> int:
        return len(self.ds)

    @property
    def lengths(self) -> List[int]:
        """Ленивое вычисление длин для ``group_by_length``."""
        if self._lengths is None:
            logger.info("Вычисление длин последовательностей …")
            self._lengths = [
                len(self.ds[i][self.cfg.prompt_column]) + len(self.ds[i][self.cfg.answer_column])
                for i in range(len(self.ds))
            ]
            logger.info(
                "Длины: min=%d, max=%d, mean=%.0f",
                min(self._lengths), max(self._lengths), np.mean(self._lengths),
            )
        return self._lengths

    def _fallback_size(self) -> tuple[int, int]:
        s = int(getattr(self.cfg, "image_size", 336))
        return (s, s)

    def _load_image(self, idx: int) -> Image.Image:
        # Режим картинки при обучении: blank/noise → неинформативный холст (текстовые
        # условия, где окрестность подаётся ТЕКСТОМ, а не картинкой).
        mode = getattr(self.cfg, "train_image_mode", "real")
        size = self._fallback_size()
        if mode == "blank":
            return Image.new("RGB", size, (255, 255, 255))
        if mode == "noise":
            arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            return Image.fromarray(arr, "RGB")

        raw_path = self.ds[int(idx)][self.cfg.image_column]
        # Собираем полный путь: image_root + относительный путь
        rel_path = raw_path.replace("\\", "/")
        full_path = str(Path(self.cfg.image_root) / rel_path) if self.cfg.image_root else rel_path

        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as exc:
            logger.warning("Битое изображение [%d]: %s — %s", idx, full_path, exc)
            img = Image.new("RGB", self._fallback_size(), _FALLBACK_COLOR)

        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.ds[int(idx)]
        image = self._load_image(idx)

        prompt = example[self.cfg.prompt_column]
        answer = example[self.cfg.answer_column]

        # Всё, что зависит от семейства модели (формат промпта, форма
        # pixel_values, image_grid_thw), инкапсулировано в families.
        return process_train_example(
            self.processor, self.tokenizer, self.family,
            prompt, answer, image, self.cfg.max_length,
        )


class DataCollatorForGraphVisLVLM:
    """Pad-коллатор: выравнивает ``input_ids / labels`` и склеивает изображения.

    Способ склейки изображений зависит от семейства (LLaVA: stack фикс.
    тензоров; Qwen: cat патчей + image_grid_thw) — делегируется families.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None,
                 family: str = "llava"):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.family = family

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return family_collate(self.family, self.tokenizer, features, self.pad_to_multiple_of)