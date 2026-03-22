from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from graphviz import Digraph
from PIL import Image

from config import VisualizationConfig

logger = logging.getLogger(__name__)


class FB15kGraphVisualizer:
    """Рендерит head-centric подграф + [?] в PNG."""

    def __init__(self, cfg: VisualizationConfig):
        self._cfg = cfg
        self._graphs_dir = Path(cfg.output_dir) / "graphs"
        self._graphs_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, str] = {} if cfg.cache_enabled else None
        self._size_inches = f"{cfg.target_size / cfg.render_dpi:.2f}"

    def render(
        self,
        subgraph: Dict[str, Any],
        query_id: str,
        target_relation: Optional[str] = None,
    ) -> str:
        safe_id = _safe_filename(query_id)
        output_path = self._graphs_dir / f"lp_{safe_id}.{self._cfg.fmt}"

        if self._cache is not None and output_path.exists():
            return str(output_path)

        dot = self._build_dot(subgraph, target_relation)
        stem = self._graphs_dir / f"lp_{safe_id}"
        dot.render(str(stem), cleanup=True)
        self._enforce_exact_size(output_path)

        result = str(output_path)
        if self._cache is not None:
            self._cache[query_id] = result
        return result

    def _enforce_exact_size(self, path: Path) -> None:
        target = self._cfg.target_size
        img = Image.open(path)
        if img.size != (target, target):
            img = img.resize((target, target), Image.LANCZOS)
            img.save(path)

    def _build_dot(
        self,
        subgraph: Dict[str, Any],
        target_relation: Optional[str],
    ) -> Digraph:
        cfg = self._cfg
        head = subgraph["head"]
        rel = subgraph.get("rel") or target_relation or "?"

        dot = Digraph(format=cfg.fmt, engine=cfg.engine)

        # ── Общие атрибуты ──
        dot.attr(
            dpi=str(cfg.render_dpi),
            size=f"{self._size_inches},{self._size_inches}!",
            ratio="fill",
            pad="0.1",
            margin="0.1",
            bgcolor="white",
        )

        # ── Атрибуты, зависящие от engine ──
        if cfg.engine == "dot":
            dot.attr(rankdir="LR", nodesep="0.3", ranksep="0.5")
        elif cfg.engine in ("neato", "fdp", "sfdp"):
            dot.attr(
                overlap="false",
                sep="+6",
                splines="curved",
            )
        elif cfg.engine == "circo":
            dot.attr(mindist="0.5")

        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fontname="Arial Bold",
            fontsize=str(cfg.node_fontsize),
            margin="0.12,0.06",
            height="0.35",
            width="0.6",
        )
        dot.attr(
            "edge",
            fontname="Arial",
            fontsize=str(cfg.edge_fontsize),
            fontcolor="gray25",
            penwidth="1.2",
            arrowsize="0.7",
            len="1.5",
        )

        # ── Узлы ──
        nodes = subgraph["nodes"][:cfg.max_nodes_display]
        edges_list = subgraph["edges"][:cfg.max_edges_display]
        allowed = {n["name"] for n in nodes}

        # Head
        dot.node(
            head,
            # _truncate(head, cfg.max_label_len),
            head,
            fillcolor="lightskyblue",
            penwidth="2.5",
            fontsize=str(cfg.head_fontsize),
        )

        # Соседи
        for node in nodes:
            name = node["name"]
            if name == head:
                continue
            dot.node(
                name,
                _truncate(name, cfg.max_label_len),
                fillcolor="lightyellow",
            )

        # ── Рёбра с подписями ──
        seen: Set[tuple] = set()
        for edge in edges_list:
            src, tgt = edge["source"], edge["target"]
            if src not in allowed or tgt not in allowed:
                continue
            key = tuple(sorted((src, tgt)))
            if key in seen:
                continue
            seen.add(key)
            dot.edge(
                src, tgt,
                label=f" {_truncate(edge['relation'], cfg.max_label_len)} ",
            )

        # ── [?] + пунктирное ребро ──
        dot.node(
            "__target__", "[?]",
            fillcolor="lightgray",
            color="darkgreen",
            style="dashed,rounded,filled",
            penwidth="2.5",
            fontcolor="darkgreen",
        )
        dot.edge(
            head, "__target__",
            label=f" {_truncate(rel, cfg.max_label_len)} ",
            style="dashed",
            color="darkgreen",
            fontcolor="darkgreen",
            penwidth="2.0",
        )

        # ── Заголовок ──
        dot.attr(
            # label=f"{_truncate(head, 18)} \u2014[{_truncate(rel, 12)}]\u2192 ?",
            label=f"{head} \u2014[{rel}]\u2192 ?",
            labelloc="t", fontsize=str(cfg.edge_fontsize + 1),
            fontname="Arial Bold",
        )
        return dot


# ── утилиты ──────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    text = text.replace("_", " ")
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space > max_len // 2:
        return cut[:last_space] + "\u2026"
    return cut + "\u2026"


def _safe_filename(text: str, max_len: int = 60) -> str:
    return text.replace("/", "_").replace("\\", "_")[:max_len]


def visualize_task(
    subgraph: Dict[str, Any],
    query_id: str,
    target_relation: Optional[str],
    cfg: VisualizationConfig,
) -> str:
    """Top-level для joblib."""
    vis = FB15kGraphVisualizer(cfg)
    return vis.render(subgraph, query_id, target_relation)