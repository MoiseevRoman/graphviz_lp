"""«Прочитай, потом ответь» (read-then-answer, RTA): целевой текст и разбор чтения.

Ответ модели при обучении:
    Edges:
    - award nominee -> Thomas Sangster      (ребро голова → сосед)
    - award winner <- 65th Academy Awards   (ребро сосед → голова)
    Answer: <хвост>
Рёбра — ровно то, что нарисовано (compute_visible_subgraph, подписи как на картинке),
поэтому чтение обучается на видимом, а не на всём KG.
"""

from __future__ import annotations

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RTA_INSTR = ("Predict the missing tail entity. First list the edges shown in the image "
             "as '- relation -> entity' lines, then write 'Answer: <entity name>'.")
LP_INSTR = "Predict the missing tail entity. Answer with the entity name only."
_LINE = re.compile(r"^-\s*(.+?)\s*(->|<-)\s*(.+?)\s*$")


def visible_pairs(head, relation, neighbors_text, max_nodes=0):
    """[(подпись ребра, подпись соседа, '->'|'<-')] как на картинке."""
    from add_text_encodings import parse_neighbors_text
    from relayout_images import _build_sg
    from config import VisualizationConfig
    from visualizer import compute_visible_subgraph, format_label
    vcfg = VisualizationConfig()
    if max_nodes > 0:
        vcfg.max_nodes_display = max_nodes
        vcfg.max_edges_display = max(vcfg.max_edges_display, max_nodes + 10)
    edges, order = parse_neighbors_text(neighbors_text)
    if not edges:
        return []
    sg = _build_sg(head, relation, edges, order, "")
    _, vis = compute_visible_subgraph(sg, vcfg)
    L = vcfg.max_label_len
    out = []
    for e in vis:
        if e["source"] == head:
            out.append((format_label(e["relation"], L), format_label(e["target"], L), "->"))
        elif e["target"] == head:
            out.append((format_label(e["relation"], L), format_label(e["source"], L), "<-"))
    return out


def reading_text(pairs):
    return "Edges:\n" + "\n".join(f"- {r} {d} {e}" for r, e, d in pairs)


def target(head, relation, neighbors_text, tail, max_nodes=0):
    return reading_text(visible_pairs(head, relation, neighbors_text, max_nodes)) + f"\nAnswer: {tail}"


def parse_reading(text):
    """Сгенерированное чтение → множество (подпись ребра, сосед, направление), нормализованное."""
    out = set()
    for ln in (text or "").split("\n"):
        m = _LINE.match(ln.strip())
        if m:
            out.add((_n(m.group(1)), _n(m.group(3)), m.group(2)))
    return out


def _n(s):
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip().lower()
