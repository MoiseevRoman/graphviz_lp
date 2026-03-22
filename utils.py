from __future__ import annotations

import re
from typing import Dict

from PIL import Image


# ────────────────────────────────────────────────────────────────
# Очистка текста
# ────────────────────────────────────────────────────────────────
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MAX_TEXT_LEN = 500


def sanitize_text(text: str, max_len: int = _MAX_TEXT_LEN) -> str:
    """Очищает строку для безопасного хранения в Neo4j / JSON."""
    if not text:
        return ""
    text = text.replace("\\", "").replace('"', "").replace("'", "")
    text = _CONTROL_CHARS.sub("", text)
    return text[:max_len].strip()


# ────────────────────────────────────────────────────────────────
# Преобразование relation path → читаемое имя
# ────────────────────────────────────────────────────────────────
_RELATION_CACHE: Dict[str, str] = {}
_VALID_START = re.compile(r"^[A-Za-z]")


def sanitize_relation_type(rel_raw: str) -> str:
    """``/film/film/genre`` → ``genre``."""
    if rel_raw in _RELATION_CACHE:
        return _RELATION_CACHE[rel_raw]

    parts = rel_raw.split("/")
    rel_type = parts[-1] if parts[-1] and _VALID_START.match(parts[-1]) else "RelatedTo"

    _RELATION_CACHE[rel_raw] = rel_type
    return rel_type


# ────────────────────────────────────────────────────────────────
# Нормализация ответа (evaluation)
# ────────────────────────────────────────────────────────────────
_ANSWER_PREFIXES = (
    "the answer is ", "answer: ", "answer is ",
    "the entity is ", "entity: ",
    "the ", "a ", "an ",
)


def normalize_answer(text: str) -> str:
    """Приводит ответ модели к каноническому виду для сравнения."""
    text = text.strip().lower()
    for prefix in _ANSWER_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()
