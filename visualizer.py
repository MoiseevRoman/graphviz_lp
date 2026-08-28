from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from graphviz import Digraph
from PIL import Image

from config import VisualizationConfig

logger = logging.getLogger(__name__)


def compute_visible_subgraph(
    subgraph: Dict[str, Any],
    cfg: VisualizationConfig,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Возвращает ровно те узлы и рёбра, которые попадут на изображение.

    Единый источник истины для рендера (``_build_dot``) и для генерации
    Stage-1 меток (counts / degrees), чтобы ответы совпадали с картинкой.
    Узел ``[?]`` и пунктирное ребро к нему сюда НЕ входят — они добавляются
    отдельно и известны обоим потребителям.
    """
    nodes = subgraph["nodes"][:cfg.max_nodes_display]
    allowed = {n["name"] for n in nodes}

    seen: Set[tuple] = set()
    visible_edges: List[Dict[str, str]] = []
    for edge in subgraph["edges"][:cfg.max_edges_display]:
        src, tgt = edge["source"], edge["target"]
        if src not in allowed or tgt not in allowed:
            continue
        key = (src, tgt, edge["relation"])  # направленный ключ
        if key in seen:
            continue
        seen.add(key)
        visible_edges.append(edge)
    return nodes, visible_edges


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

    def layout_positions(
        self,
        subgraph: Dict[str, Any],
        target_relation: Optional[str] = None,
    ) -> Dict[str, tuple]:
        """Координаты узлов, ВЫЧИСЛЕННЫЕ ТЕМ ЖЕ движком, что рисует картинку.

        Это и есть «инженерная работа» Graphviz, которую движок впечатывает в
        изображение (центральность через положение, кластеры через близость).
        Извлекаем её явно, чтобы подать в текст и проверить: даёт ли выигрыш
        картинки именно эта предвычисленная раскладка, а не визуальная модальность.

        Возвращает ``{raw_name: (x, y)}`` в нормированных координатах [0, 1]
        (y снизу вверх). Ключ узла-заглушки — ``"[?]"``. Пустой dict, если
        бинарь ``dot`` недоступен или layout не распарсился.
        """
        import json as _json

        dot = self._build_dot(subgraph, target_relation)
        # Обратное соответствие DOT-идентификатор → сырое имя.
        nodes, _ = compute_visible_subgraph(subgraph, self._cfg)
        id2raw = {_nid(n["name"]): n["name"] for n in nodes}
        id2raw[_nid(subgraph["head"])] = subgraph["head"]
        id2raw["__target__"] = "[?]"

        try:
            data = _json.loads(dot.pipe(format="json").decode("utf-8"))
        except Exception as exc:
            logger.warning("layout_positions: не удалось получить json от dot: %s", exc)
            return {}

        bb = data.get("bb", "")
        try:
            x0, y0, x1, y1 = (float(v) for v in bb.split(","))
            w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        except Exception:
            x0 = y0 = 0.0
            w = h = 1.0

        out: Dict[str, tuple] = {}
        for obj in data.get("objects", []):
            pos = obj.get("pos")
            name = obj.get("name")
            if not pos or name is None:
                continue
            raw = id2raw.get(name, name)
            try:
                px, py = (float(v) for v in pos.split(","))
            except Exception:
                continue
            out[raw] = ((px - x0) / w, (py - y0) / h)
        return out

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
        rel = _norm_rel(subgraph, target_relation)

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

        # ── Узлы и рёбра (ровно то, что видно) ──
        nodes, edges_list = compute_visible_subgraph(subgraph, cfg)

        # Head
        dot.node(
            _nid(head),
            format_label(head, cfg.max_label_len),
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
                _nid(name),
                format_label(name, cfg.max_label_len),
                fillcolor="lightyellow",
            )

        # ── Рёбра с подписями (уже отфильтрованы и дедуплицированы) ──
        for edge in edges_list:
            dot.edge(
                _nid(edge["source"]), _nid(edge["target"]),
                label=f" {format_label(edge['relation'], cfg.max_label_len)} ",
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
            _nid(head), "__target__",
            label=f" {format_label(rel, cfg.max_label_len)} ",
            style="dashed",
            color="darkgreen",
            fontcolor="darkgreen",
            penwidth="2.0",
        )

        # ── Заголовок ──
        dot.attr(
            label=(
                f"{format_label(head, cfg.max_label_len)} "
                f"\u2014[{format_label(rel, cfg.max_label_len)}]\u2192 ?"
            ),
            labelloc="t", fontsize=str(cfg.edge_fontsize + 1),
            fontname="Arial Bold",
        )
        return dot


# ── утилиты ──────────────────────────────────────────────────

def _norm_rel(subgraph: Dict[str, Any], target_relation: Optional[str] = None) -> str:
    """Единая нормализация метки target-ребра для рендера и текста.

    Пустая строка трактуется как отсутствие → fallback на target_relation,
    затем на '?'. Гарантирует совпадение подписи head→[?] между модальностями.
    """
    return (subgraph.get("rel") or "").strip() or (target_relation or "") or "?"


def _nid(name: str) -> str:
    """Безопасный DOT-идентификатор узла.

    Питоновский graphviz трактует ':' в конечной точке РЕБРА как разделитель
    node:port (`"A:b" -> X` → узел "A", порт "b"). Из-за этого рёбра к
    сущностям с двоеточием в имени (напр. 'Star_Trek:_Deep_Space_Nine')
    привязывались к фантомному пустому узлу, а настоящий подписанный бокс
    оставался без ребра. Экранируем ':' в ИДЕНТИФИКАТОРЕ (метки не трогаем).
    """
    return name.replace(":", "：") if ":" in name else name


def format_label(name: str, max_len: int) -> str:
    """\u0415\u0434\u0438\u043d\u0430\u044f \u041f\u041e\u0412\u0415\u0420\u0425\u041d\u041e\u0421\u0422\u041d\u0410\u042f \u0424\u041e\u0420\u041c\u0410 \u0438\u043c\u0435\u043d\u0438 \u0441\u0443\u0449\u043d\u043e\u0441\u0442\u0438/\u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u044f.

    \u0415\u0434\u0438\u043d\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a \u0442\u043e\u0433\u043e, \u041a\u0410\u041a \u043f\u043e\u0434\u043f\u0438\u0441\u0430\u043d \u0443\u0437\u0435\u043b/\u0440\u0435\u0431\u0440\u043e. \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442\u0441\u044f \u0438 \u0432
    \u0440\u0435\u043d\u0434\u0435\u0440\u0435 (visualizer), \u0438 \u0432\u043e \u0432\u0441\u0435\u0445 \u0442\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0445 \u043f\u0440\u0435\u0434\u0441\u0442\u0430\u0432\u043b\u0435\u043d\u0438\u044f\u0445 (Stage-1 \u043e\u0442\u0432\u0435\u0442\u044b,
    text_encoders), \u0447\u0442\u043e\u0431\u044b \u043c\u043e\u0434\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u0438 \u0431\u044b\u043b\u0438 \u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u043e\u043d\u043d\u043e \u042d\u041a\u0412\u0418\u0412\u0410\u041b\u0415\u041d\u0422\u041d\u042b \u043f\u043e
    \u043d\u0430\u043f\u0438\u0441\u0430\u043d\u0438\u044e \u0438\u043c\u0451\u043d (\u0438\u043d\u0430\u0447\u0435 \u043a\u0430\u0440\u0442\u0438\u043d\u043a\u0430 \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u0442 'Star Trek Deep Space', \u0430 \u0442\u0435\u043a\u0441\u0442
    \u0442\u0440\u0435\u0431\u0443\u0435\u0442 'Star_Trek:_Deep_Space_Nine' \u2014 \u0441\u043a\u0440\u044b\u0442\u044b\u0439 \u043a\u043e\u043d\u0444\u0430\u0443\u043d\u0434 \u0441\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u044f).
    """
    return _truncate(name, max_len)


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