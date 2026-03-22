from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from graphvis_lp.config import TrainConfig

logger = logging.getLogger(__name__)

_FALLBACK_SIZE = (336, 336)
_FALLBACK_COLOR = (128, 128, 128)


class GraphVisLVLMDataset(TorchDataset):
    """Dataset, который на лету строит ``(input_ids, labels, pixel_values)``."""

    def __init__(self, hf_dataset, processor, tokenizer, cfg: TrainConfig):
        self.ds = hf_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._image_token: str = getattr(processor, "image_token", "<image>")
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

    def _load_image(self, idx: int) -> Image.Image:
        raw_path = self.ds[int(idx)][self.cfg.image_column]
        # Собираем полный путь: image_root + относительный путь
        rel_path = raw_path.replace("\\", "/")
        full_path = str(Path(self.cfg.image_root) / rel_path) if self.cfg.image_root else rel_path

        try:
            img = Image.open(full_path).convert("RGB")
        except Exception as exc:
            logger.warning("Битое изображение [%d]: %s — %s", idx, full_path, exc)
            img = Image.new("RGB", _FALLBACK_SIZE, _FALLBACK_COLOR)

        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.ds[int(idx)]
        image = self._load_image(idx)

        prompt = example[self.cfg.prompt_column].strip()
        answer = example[self.cfg.answer_column].strip()

        input_text = (
            prompt if self._image_token in prompt else f"{self._image_token}\n{prompt}"
        )
        full_text = f"{input_text} {answer}{self.tokenizer.eos_token}"

        processed = self.processor(
            text=full_text, images=image, return_tensors="pt",
            padding=False, truncation=True, max_length=self.cfg.max_length,
        )
        input_ids = processed["input_ids"][0]
        attention_mask = processed["attention_mask"][0]

        # Длина промпта для маскирования лейблов
        prompt_enc = self.processor(
            text=input_text, images=image, return_tensors="pt",
            padding=False, truncation=True, max_length=self.cfg.max_length,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": processed["pixel_values"][0],
            "labels": labels,
        }


class DataCollatorForGraphVisLVLM:
    """Pad-коллатор: выравнивает ``input_ids / labels`` по максимальной длине."""

    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels_list = [f["labels"] for f in features]
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        max_len = max(ids.size(0) for ids in input_ids)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = (max_len + m - 1) // m * m

        pad_id = self.tokenizer.pad_token_id

        def _pad(tensor: torch.Tensor, fill: int) -> torch.Tensor:
            n = max_len - tensor.size(0)
            return torch.cat([tensor, tensor.new_full((n,), fill)]) if n > 0 else tensor

        return {
            "input_ids": torch.stack([_pad(ids, pad_id) for ids in input_ids]),
            "attention_mask": torch.stack([_pad(m, 0) for m in attention_mask]),
            "labels": torch.stack([_pad(lb, -100) for lb in labels_list]),
            "pixel_values": pixel_values,
        }