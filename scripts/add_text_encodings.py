#!/usr/bin/env python3
"""Добавляет альтернативные текстовые кодировки подграфа в готовые датасеты.

Мотивация: для ЧИСТОГО сравнения модальностей нужен сильнейший текстовый
бейзлайн. Одна и та же окрестность (те же рёбра, что в neighbors_text и на
картинке) сериализуется разными способами — edge-list / adjacency-list /
adjacency-matrix. Здесь мы НЕ пере-рендерим и НЕ ходим в Neo4j: рёбра берём
из уже готовой колонки neighbors_text, поэтому информация гарантированно
совпадает с условиями text/visual.

Добавляемые колонки: neighbors_adjlist, neighbors_matrix
(edge-list уже лежит в neighbors_text → context_mode=text).

Колонки только ДОБАВЛЯЮТСЯ (существующие не меняются), операция идемпотентна.
Запись на месте через временную папку + атомарную замену.

Пример:
    python scripts/add_text_encodings.py --budgets 1,4,8,16 \
        --data_base graphvis_scaling --encodings adjlist,matrix
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from text_encoders import _encode_adjlist, _encode_matrix  # noqa: E402

_LINE = re.compile(r"^- (.*?) --(.*?)--> (.*)$")
_ENCODERS = {"adjlist": _encode_adjlist, "matrix": _encode_matrix}


def parse_neighbors_text(nt: str):
    """'- A --rel words--> B' → (edges[(s,r,t)], node_labels[head-first])."""
    edges, order = [], []
    for ln in (nt or "").split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        m = _LINE.match(ln)
        if not m:
            return None, None  # неожиданный формат → пропускаем строку-источник
        s, r, t = m.group(1), m.group(2), m.group(3)
        for n in (s, t):
            if n not in order:
                order.append(n)
        edges.append((s, r, t))
    return edges, order


def _row_cols(nt: str, kinds):
    edges, order = parse_neighbors_text(nt)
    if edges is None:
        # громкий отказ: тихий откат к edge-list испортил бы matched-info
        raise ValueError(f"нераспарсенный neighbors_text для кодировки: {nt!r:.200}")
    return {f"neighbors_{k}": _ENCODERS[k](order, edges) for k in kinds}


def process_split(path: str, kinds, force: bool) -> str:
    from datasets import load_from_disk
    ds = load_from_disk(path)
    have = set(ds.column_names)
    todo = [k for k in kinds if force or f"neighbors_{k}" not in have]
    if not todo:
        return f"{path}: уже есть {['neighbors_'+k for k in kinds]} — пропуск"
    if "neighbors_text" not in have:
        return f"{path}: НЕТ neighbors_text — пропуск"
    # если force и колонка есть — удалим, чтобы map не конфликтовал
    for k in todo:
        if f"neighbors_{k}" in ds.column_names:
            ds = ds.remove_columns(f"neighbors_{k}")

    ds = ds.map(lambda r: _row_cols(r["neighbors_text"], todo),
                desc=f"encode[{','.join(todo)}]",
                keep_in_memory=True, load_from_cache_file=False)

    tmp = path + ".tmp_enc"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    ds.save_to_disk(tmp)
    del ds
    bak = path + ".bak_enc"
    if os.path.isdir(bak):
        shutil.rmtree(bak)
    os.rename(path, bak)
    os.rename(tmp, path)
    shutil.rmtree(bak)
    return f"{path}: добавлено {['neighbors_'+k for k in todo]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="1,4,8,16", help="K через запятую")
    ap.add_argument("--data_base", default=os.path.join(_ROOT, "graphvis_scaling"))
    ap.add_argument("--splits", default="stage2_train,stage2_test")
    ap.add_argument("--encodings", default="adjlist,matrix")
    ap.add_argument("--force", action="store_true", help="перезаписать существующие колонки")
    ap.add_argument("--dry_run", action="store_true", help="только показать пример кодировок")
    args = ap.parse_args()

    kinds = [k.strip() for k in args.encodings.split(",") if k.strip()]
    bad = [k for k in kinds if k not in _ENCODERS]
    if bad:
        sys.exit(f"неизвестные кодировки: {bad}; доступно: {list(_ENCODERS)}")
    Ks = [int(x) for x in args.budgets.split(",") if x.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.dry_run:
        sample = ("- A --directed--> B\n- A --directed--> C\n- D --spouse of--> A")
        print("Пример входа:\n" + sample)
        for k, v in _row_cols(sample, kinds).items():
            print(f"\n--- {k} ---\n{v}")
        return

    for k in Ks:
        for sp in splits:
            path = os.path.join(args.data_base, f"k{k}", "datasets", sp)
            if not os.path.isdir(path):
                print(f"[k={k}] {sp}: нет папки {path} — пропуск")
                continue
            print("[k=%d] %s" % (k, process_split(path, kinds, args.force)))


if __name__ == "__main__":
    main()
