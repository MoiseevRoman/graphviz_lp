"""Текстовые кодировки head-centric подграфа для сравнения модальностей.

Ключевой принцип чистоты эксперимента: текст и картинка должны нести
ИДЕНТИЧНУЮ информацию. Поэтому все кодировки строятся из ТОГО ЖЕ источника,
что и рендер (``compute_visible_subgraph``), и через ту же поверхностную форму
имён (``format_label``). То, что видит модель на картинке, и то, что она читает
в тексте, — один и тот же набор узлов/рёбер с одинаковыми подписями.

Что попадает в кодировку (ровно как на изображении):
    • видимые узлы + узел-заглушка ``[?]``;
    • видимые рёбра + пунктирное ребро ``head —[rel]→ [?]``.
Tail нигде не фигурирует (он вырезан ещё в retriever) → leak-free.

Три кодировки (разные способы подать ОДИН граф):
    • ``triples``  — список триплетов (source, relation, target);
    • ``adjlist``  — список смежности (по исходящим рёбрам);
    • ``matrix``   — матрица смежности 0/1 + легенда рёбер с отношениями.

Инъекция инженерных признаков движка (degree/centrality/layout) — отдельный
модуль (пункт 3 плана), здесь только «сырые» представления структуры.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from config import VisualizationConfig
from graph_features import compute_features
from visualizer import FB15kGraphVisualizer, compute_visible_subgraph, format_label

QUESTION_NODE = "[?]"

_KIND_DESC = {
    "triples": "a list of (source, relation, target) triples",
    "adjlist": "an adjacency list (outgoing edges per node)",
    "matrix": "an adjacency matrix with an edge-relation legend",
    "feat": "per-node structural features precomputed by the graph layout engine",
    "triples_feat": "a list of triples plus per-node engine-computed features",
}


# ────────────────────────────────────────────────────────────────
# Единый видимый граф (зеркало картинки)
# ────────────────────────────────────────────────────────────────
def _visible(
    subgraph: Dict[str, Any], cfg: VisualizationConfig,
) -> Tuple[str, str, List[str], List[Tuple[str, str, str]]]:
    """→ (head_label, rel_label, node_labels[incl [?]], edges[(s,r,t) labels]).

    ``node_labels`` — по позиции соответствуют сырым узлам (для матрицы); рёбра
    отформатированы и включают пунктирное head→[?], как на изображении.
    """
    nodes, edges = compute_visible_subgraph(subgraph, cfg)
    head = subgraph["head"]
    rel = subgraph.get("rel", "?")
    mll = cfg.max_label_len

    def lbl(name: str) -> str:
        return name if name == QUESTION_NODE else format_label(name, mll)

    raw_nodes = [n["name"] for n in nodes] + [QUESTION_NODE]
    node_labels = [lbl(n) for n in raw_nodes]

    fmt_edges: List[Tuple[str, str, str]] = [
        (lbl(e["source"]), format_label(e["relation"], mll), lbl(e["target"]))
        for e in edges
    ]
    # Пунктирное ребро к цели — присутствует на картинке, значит и в тексте.
    fmt_edges.append((lbl(head), format_label(rel, mll), QUESTION_NODE))

    return lbl(head), format_label(rel, mll), node_labels, fmt_edges


# ────────────────────────────────────────────────────────────────
# Кодировки
# ────────────────────────────────────────────────────────────────
def _encode_triples(node_labels: List[str], edges: List[Tuple[str, str, str]]) -> str:
    lines = [f"({s}, {r}, {t})" for s, r, t in edges]
    return "\n".join(lines)


def _encode_adjlist(node_labels: List[str], edges: List[Tuple[str, str, str]]) -> str:
    # Группировка по исходящему узлу; порядок узлов — как в node_labels, чтобы
    # инвентарь вершин совпадал с картинкой (в т.ч. изолированные показаны).
    out: Dict[str, List[str]] = {n: [] for n in node_labels}
    for s, r, t in edges:
        out.setdefault(s, []).append(f"{r} -> {t}")
    lines = []
    for n in node_labels:
        adj = out.get(n, [])
        lines.append(f"{n}: " + ("; ".join(adj) if adj else "(no outgoing edges)"))
    return "\n".join(lines)


def _encode_matrix(node_labels: List[str], edges: List[Tuple[str, str, str]]) -> str:
    idx = {n: i for i, n in enumerate(node_labels)}
    size = len(node_labels)
    mat = [[0] * size for _ in range(size)]
    legend: List[str] = []
    for s, r, t in edges:
        i, j = idx.get(s), idx.get(t)
        if i is None or j is None:
            continue
        mat[i][j] = 1
        legend.append(f"({i},{j}): {r}")

    header = "Node index:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(node_labels))
    rows = ["Adjacency matrix (row=source, col=target, 1=edge):"]
    col_hdr = "     " + " ".join(f"{i:>2}" for i in range(size))
    rows.append(col_hdr)
    for i in range(size):
        rows.append(f"  {i:>2} " + " ".join(f"{mat[i][j]:>2}" for j in range(size)))
    leg = "Edge relations:\n" + "\n".join(f"  {x}" for x in legend) if legend else ""
    return "\n".join([header, "", *rows] + (["", leg] if leg else []))


_ENCODERS = {
    "triples": _encode_triples,
    "adjlist": _encode_adjlist,
    "matrix": _encode_matrix,
}


# ────────────────────────────────────────────────────────────────
# Feature-блок (инженерные признаки движка → текст)
# ────────────────────────────────────────────────────────────────
def _format_feat_lines(feats, cfg: VisualizationConfig) -> str:
    """Форматирует уже вычисленные признаки в текстовый блок."""
    mll = cfg.max_label_len

    def lbl(name: str) -> str:
        return name if name == QUESTION_NODE else format_label(name, mll)

    lines = []
    for raw, f in feats.items():
        parts = [
            f"degree={int(f['degree'])}",
            f"pagerank={f['pagerank']:.3f}",
            f"betweenness={f['betweenness']:.3f}",
        ]
        if "x" in f:
            parts.append(f"pos=({f['x']:.2f},{f['y']:.2f})")
        lines.append(f"{lbl(raw)}: " + ", ".join(parts))
    return "\n".join(lines)


def encode_features_block(
    subgraph: Dict[str, Any], cfg: VisualizationConfig, include_layout: bool,
) -> str:
    """Текстовая сводка признаков на узел (degree/центральность[/позиция]).

    Ровно те свойства, что движок неявно кодирует в картинке. Подписи узлов — в
    той же поверхностной форме, что на изображении и в остальных кодировках.
    """
    feats = compute_features(subgraph, cfg, include_layout=include_layout)
    return _format_feat_lines(feats, cfg)


def encode_subgraph(subgraph: Dict[str, Any], cfg: VisualizationConfig, kind: str) -> str:
    """Текстовый блок с окрестностью в кодировке ``kind`` (без обёртки-промпта).

    ``kind`` ∈ {triples, adjlist, matrix, feat, triples_feat}. Варианты с
    признаками используют ``cfg.inject_layout_features`` для добавления позиций.
    """
    if kind in _ENCODERS:
        _, _, node_labels, edges = _visible(subgraph, cfg)
        return _ENCODERS[kind](node_labels, edges)

    inc_layout = bool(getattr(cfg, "inject_layout_features", False))
    if kind == "feat":
        return encode_features_block(subgraph, cfg, inc_layout)
    if kind == "triples_feat":
        _, _, node_labels, edges = _visible(subgraph, cfg)
        block = _encode_triples(node_labels, edges)
        feat = encode_features_block(subgraph, cfg, inc_layout)
        return f"{block}\n\nNode features:\n{feat}"
    raise ValueError(f"Unknown encoding kind: {kind!r}")


# ────────────────────────────────────────────────────────────────
# Полный leak-free текстовый промпт (аналог Stage-2, но без <image>)
# ────────────────────────────────────────────────────────────────
def build_text_prompt(
    subgraph: Dict[str, Any],
    cfg: VisualizationConfig,
    kind: str,
    head_description: str = "",
) -> str:
    """Текстовый промпт link prediction. Единственное отличие от картиночного —
    окрестность подана ТЕКСТОМ (кодировка ``kind``), а не изображением. Формулы
    запроса/инструкции держим максимально близкими к Stage-2 картинки, чтобы
    сравнение измеряло именно модальность подачи графа, а не разницу wording.
    """
    head_label, rel_label, _, _ = _visible(subgraph, cfg)
    block = encode_subgraph(subgraph, cfg, kind)
    desc = head_description or head_label
    return (
        "Knowledge Graph Link Prediction\n\n"
        f"The neighborhood of entity '{head_label}' is encoded below as "
        f"{_KIND_DESC[kind]}. The target entity to predict is denoted {QUESTION_NODE}.\n\n"
        f"{block}\n\n"
        f"Entity description: {desc}\n\n"
        f"Query: ({head_label}, {rel_label}, ?)\n"
        "Based on the graph structure, predict the missing tail entity.\n"
        "Provide only the entity name.\n"
        "Answer:"
    )


# Колонки, эмитируемые для датасета. feat / triples_feat — ветка гипотезы
# «текст + инженерные признаки движка» (пункт 3).
_PROMPT_KINDS = ("triples", "adjlist", "matrix", "feat", "triples_feat")


def all_text_prompts(
    subgraph: Dict[str, Any],
    cfg: VisualizationConfig,
    head_description: str = "",
) -> Dict[str, str]:
    """→ {'prompt_triples':..., 'prompt_adjlist':..., 'prompt_matrix':...,
          'prompt_feat':..., 'prompt_triples_feat':...}.

    Готовые колонки промптов для датасета: выбираются в evaluate через
    ``prompt_key`` и в обучении текстовых условий (train_image_mode='blank').
    Сравнение ``prompt_triples`` (чистый текст) vs ``prompt_triples_feat``
    (текст + признаки движка) и есть проверка гипотезы про indutive bias.
    """
    # Признаки (и, возможно, layout через dot) считаем ОДИН раз на пример —
    # переиспользуем для feat и triples_feat, чтобы не звать dot дважды.
    head_label, rel_label, node_labels, edges = _visible(subgraph, cfg)
    inc_layout = bool(getattr(cfg, "inject_layout_features", False))
    vis = FB15kGraphVisualizer(cfg) if inc_layout else None
    feats = compute_features(subgraph, cfg, include_layout=inc_layout, visualizer=vis)
    feat_block = _format_feat_lines(feats, cfg)
    triples_block = _encode_triples(node_labels, edges)

    blocks = {
        "triples": triples_block,
        "adjlist": _encode_adjlist(node_labels, edges),
        "matrix": _encode_matrix(node_labels, edges),
        "feat": feat_block,
        "triples_feat": f"{triples_block}\n\nNode features:\n{feat_block}",
    }
    desc = head_description or head_label

    def _wrap(kind: str, block: str) -> str:
        return (
            "Knowledge Graph Link Prediction\n\n"
            f"The neighborhood of entity '{head_label}' is encoded below as "
            f"{_KIND_DESC[kind]}. The target entity to predict is denoted {QUESTION_NODE}.\n\n"
            f"{block}\n\n"
            f"Entity description: {desc}\n\n"
            f"Query: ({head_label}, {rel_label}, ?)\n"
            "Based on the graph structure, predict the missing tail entity.\n"
            "Provide only the entity name.\n"
            "Answer:"
        )

    return {f"prompt_{kind}": _wrap(kind, block) for kind, block in blocks.items()}
