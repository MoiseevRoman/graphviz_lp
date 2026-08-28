"""Инженерные признаки, которые движок отрисовки впечатывает в картинку.

Гипотеза (из обсуждения с научруком): выигрыш картинки над плоским текстом
может объясняться не визуальной модальностью, а тем, что Graphviz ПРЕДВЫЧИСЛЯЕТ
структуру — ставит хабы в центр, разносит кластеры, кодирует центральность
положением. Если подать эти же признаки ТЕКСТОМ и текст догонит картинку —
значит преимущество даёт предвычисление, а не «зрение».

Здесь мы вычисляем ровно те свойства, которые на изображении считываются
неявно:
    • degree / in / out — «хабность» (на картинке — число сходящихся линий);
    • pagerank, betweenness — центральность (на картинке — положение к центру);
    • layout-координаты — буквально то, куда движок поставил узел.

Всё считается по ТОМУ ЖЕ видимому подграфу (incl. узел [?] и пунктирное ребро
head→[?]), что и рендер, — иначе признаки не соответствовали бы картинке.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import networkx as nx

from config import VisualizationConfig
from visualizer import FB15kGraphVisualizer, compute_visible_subgraph

QUESTION_NODE = "[?]"


def _build_nx(subgraph: Dict[str, Any], cfg: VisualizationConfig):
    """Направленный граф видимой окрестности (как на картинке) + узел [?]."""
    nodes, edges = compute_visible_subgraph(subgraph, cfg)
    head = subgraph["head"]
    rel = subgraph.get("rel", "?")

    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["name"])
    g.add_node(QUESTION_NODE)
    for e in edges:
        g.add_edge(e["source"], e["target"], relation=e["relation"])
    # Пунктирное ребро к цели — присутствует на изображении.
    g.add_edge(head, QUESTION_NODE, relation=rel)
    return g


def structural_features(
    subgraph: Dict[str, Any], cfg: VisualizationConfig,
) -> "OrderedDict[str, Dict[str, float]]":
    """→ raw_name → {degree, in_degree, out_degree, pagerank, betweenness}.

    Центральности считаем на НЕОРИЕНТИРОВАННОЙ проекции: визуальная «хабность»
    на картинке симметрична (сила пружин не зависит от направления стрелки).
    """
    g = _build_nx(subgraph, cfg)
    und = g.to_undirected()

    try:
        pr = nx.pagerank(und, max_iter=200)
    except Exception:
        pr = {n: 0.0 for n in und}
    try:
        bt = nx.betweenness_centrality(und, normalized=True)
    except Exception:
        bt = {n: 0.0 for n in und}

    feats: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    for n in g.nodes():
        feats[n] = {
            "degree": float(und.degree(n)),
            "in_degree": float(g.in_degree(n)),
            "out_degree": float(g.out_degree(n)),
            "pagerank": float(pr.get(n, 0.0)),
            "betweenness": float(bt.get(n, 0.0)),
        }
    return feats


def layout_features(
    subgraph: Dict[str, Any],
    cfg: VisualizationConfig,
    visualizer: Optional[FB15kGraphVisualizer] = None,
) -> Dict[str, tuple]:
    """Координаты узлов от РЕАЛЬНОГО движка рендера. {} если dot недоступен."""
    vis = visualizer or FB15kGraphVisualizer(cfg)
    return vis.layout_positions(subgraph, subgraph.get("rel"))


def compute_features(
    subgraph: Dict[str, Any],
    cfg: VisualizationConfig,
    include_layout: bool = False,
    visualizer: Optional[FB15kGraphVisualizer] = None,
) -> "OrderedDict[str, Dict[str, float]]":
    """Полный набор признаков на узел. ``include_layout`` добавляет x/y от движка
    (дороже — вызывает dot на подграф). Порядок узлов = порядок на картинке."""
    feats = structural_features(subgraph, cfg)
    if include_layout:
        pos = layout_features(subgraph, cfg, visualizer)
        for name, xy in pos.items():
            if name in feats:
                feats[name]["x"] = float(xy[0])
                feats[name]["y"] = float(xy[1])
    return feats
