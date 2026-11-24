"""Candidate-aware подграф с путями head → x → candidate (без Neo4j).

Зачем: head-centric звезда не содержит структурного сигнала о хвосте — в
FB15k-237 правильный хвост никогда не соседствует с головой напрямую (0 %), но в
~74 % тест-троек лежит ровно в 2 хопах. Здесь на картинку/в текст попадают голова,
N кандидатов и промежуточные узлы x на путях длины 2 — то, что эвристики вроде
common neighbors используют для ранжирования.

Соглашения те же, что в основном конвейере:
  • идентичность сущностей — display-имя (как в Neo4j-графе), отношения —
    последний сегмент пути (_lastseg);
  • кандидаты — топ-cap хвостов отношения по частоте в train + золотой
    (как textlp_upperbound.make_candidates / viability_gate);
  • контекст по сплитам: train-запросы видят train, valid — train, test — train+valid;
    целевое ребро (h, r, gold) и обратное к нему исключаются всегда.

Защита от утечки: золотой хвост НИКОГДА не добавляется в показанные кандидаты
принудительно — он попадает туда только если его выбрал предотбор (freq / cn_freq),
который золота не видит, из пула top_candidates — БЕЗ золотого (см. top_candidates).
Список кандидатов не фильтруется по меткам (см. build). Порядок кандидатов в тексте — алфавитный, чтобы текст не
нёс ранга предотбора, которого нет на картинке.

Matched information: текст и картинка строятся из ОДНОГО объекта PathSubgraph,
флаги label_intermediates / label_edges действуют на обе модальности одинаково.
"""

from __future__ import annotations

import collections
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

_SPLIT_FILES = {
    "train": ["train.txt"],
    "valid": ["train.txt"],
    "test": ["train.txt", "valid.txt"],
}


def _lastseg(r: str) -> str:
    return r.split("/")[-1] or r


@dataclass
class PathSubgraphConfig:
    n_show: int = 10                # сколько кандидатов попадает на картинку/в текст
    max_paths_per_cand: int = 3     # сколько промежуточных x на кандидата
    max_rels_per_pair: int = 1      # сколько подписей отношений между парой узлов
    preselect: str = "freq"         # freq | cn_freq — как выбирать n_show из cand_cap
    cand_cap: int = 50
    label_intermediates: bool = False   # False → x рисуются точками, в тексте x1, x2…
    label_edges: bool = True
    # Для train-запросов: убрать ВСЕ прямые рёбра head–gold (а не только целевое).
    # В тесте FB15k-237 пары (h, t) не связаны в train по построению датасета; без
    # этого флага модель выучила бы «прямое ребро ⇒ ответ», которого в тесте нет.
    drop_gold_direct: bool = False


@dataclass
class PathSubgraph:
    head: str
    relation: str
    shown: List[str]                                    # показанные кандидаты
    inter: List[str]                                    # промежуточные узлы (порядок = id)
    # (src, rel, dst) в ИСТИННОМ направлении ребра
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    # кандидат → число путей длины 2 в полном графе (не ограничено max_paths)
    cn_full: Dict[str, int] = field(default_factory=dict)
    # кандидат → число показанных путей (≤ max_paths_per_cand)
    cn_shown: Dict[str, int] = field(default_factory=dict)
    direct: Dict[str, int] = field(default_factory=dict)  # кандидат → прямых рёбер с head
    # кандидат → показанные посредники в порядке специфичности (для обрезки по бюджету)
    paths: Dict[str, List[str]] = field(default_factory=dict)


class PathGraph:
    """Name-level граф FB15k-237 для одного режима сплита + частоты отношений."""

    def __init__(self, data_dir: str, split: str = "test"):
        self.data_dir = data_dir
        self.mid2name = self._load_names(data_dir)
        # (a, b) → [(rel, направление)], где направление True = a→b
        self.pair_rels: Dict[Tuple[str, str], List[Tuple[str, bool]]] = collections.defaultdict(list)
        self.adj: Dict[str, Set[str]] = collections.defaultdict(set)
        for fn in _SPLIT_FILES[split]:
            for h, r, t in self._read(fn):
                if h == t:
                    continue
                self.adj[h].add(t)
                self.adj[t].add(h)
                self.pair_rels[(h, t)].append((r, True))
                self.pair_rels[(t, h)].append((r, False))
        self.rel2tails: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.hr2tails: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
        for fn in ("train.txt", "valid.txt", "test.txt"):
            for h, r, t in self._read(fn):
                self.hr2tails[(h, r)].add(t)
                if fn == "train.txt":
                    self.rel2tails[r][t] += 1

    @staticmethod
    def _load_names(data_dir: str) -> Dict[str, str]:
        out = {}
        for line in open(os.path.join(data_dir, "FB15k_mid2name.txt"), encoding="utf-8"):
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                out[a[0]] = a[1].strip()
        return out

    def name(self, mid: str) -> str:
        return self.mid2name.get(mid, mid.split("/")[-1].replace("_", " "))

    def _read(self, fn: str):
        fp = os.path.join(self.data_dir, fn)
        if not os.path.exists(fp):
            return
        for line in open(fp, encoding="utf-8"):
            a = line.strip().split("\t")
            if len(a) == 3:
                yield self.name(a[0]), _lastseg(a[1]), self.name(a[2])

    # ── кандидаты ──
    def top_candidates(self, rel: str, cap: int) -> List[str]:
        """Пул для КОНТЕКСТА: топ-cap хвостов отношения по частоте, БЕЗ золотого.

        Золото, добавленное в пул «51-м», при показе в контексте выдаёт ответ
        (редкая сущность, которой неоткуда взяться) — так было в graphvis_paths v1.
        """
        cnt = self.rel2tails.get(rel)
        return [t for t, _ in cnt.most_common(cap)] if cnt else []

    def candidates(self, rel: str, gold: str, cap: int) -> List[str]:
        """Пул для РАНЖИРОВАНИЯ при оценке (как textlp_upperbound.make_candidates):
        топ-cap + золотой. В контекст не идёт — модель этот список не видит."""
        cnt = self.rel2tails.get(rel)
        cands = [t for t, _ in cnt.most_common(cap)] if cnt else []
        if gold not in cands:
            cands.append(gold)
        return cands

    def filtered_out(self, head: str, rel: str, gold: str) -> Set[str]:
        return self.hr2tails.get((head, rel), set()) - {gold}

    # ── структура между head и кандидатом ──
    def _neighbors(self, node: str, head: str, rel: str, gold: str,
                   drop_direct: bool = False) -> Set[str]:
        """Соседи без целевого ребра (h, r, gold) в любом направлении."""
        nb = self.adj.get(node, set())
        if node == head and gold in nb and (drop_direct or self._only_target(head, gold, rel)):
            nb = nb - {gold}
        return nb

    def _only_target(self, a: str, b: str, rel: str) -> bool:
        return all(r == rel for r, _ in self.pair_rels.get((a, b), []))

    def _pair_edges(self, a: str, b: str, head: str, rel: str, gold: str,
                    drop_direct: bool = False) -> List[Tuple[str, str, str]]:
        out = []
        for r, fwd in self.pair_rels.get((a, b), []):
            if {a, b} == {head, gold} and (drop_direct or r == rel):
                continue
            out.append((a, r, b) if fwd else (b, r, a))
        return out

    def common(self, head: str, cand: str, rel: str, gold: str,
               drop_direct: bool = False) -> Set[str]:
        nh = self._neighbors(head, head, rel, gold, drop_direct)
        return (nh & self.adj.get(cand, set())) - {head, cand}

    def degree(self, node: str) -> int:
        return len(self.adj.get(node, ()))

    def build(self, head: str, rel: str, gold: str, cfg: PathSubgraphConfig,
              cands: Optional[List[str]] = None) -> PathSubgraph:
        if cands is None:
            cands = self.top_candidates(rel, cfg.cand_cap)
        # Пул НЕ фильтруется по другим истинным хвостам: это знание из меток
        # (в т.ч. тестовых), вход модели от него зависеть не должен. Фильтрация —
        # только при подсчёте ранга, как в training/rank_eval.
        dd = cfg.drop_gold_direct
        pool = [c for c in cands if c != head]

        cn_full = {c: len(self.common(head, c, rel, gold, dd)) for c in pool}
        freq = self.rel2tails.get(rel, collections.Counter())
        if cfg.preselect == "freq":
            key = lambda c: (-freq[c], c)
        elif cfg.preselect == "cn_freq":
            key = lambda c: (-cn_full[c], -freq[c], c)
        else:
            raise ValueError(cfg.preselect)
        shown = sorted(pool, key=key)[:cfg.n_show]

        sg = PathSubgraph(head=head, relation=rel, shown=sorted(shown), inter=[])
        shown_set = set(sg.shown)
        inter_set: Dict[str, None] = {}
        seen_edges: Set[Tuple[str, str, str]] = set()

        def add_pair(a: str, b: str) -> None:
            for e in self._pair_edges(a, b, head, rel, gold, dd)[:cfg.max_rels_per_pair]:
                if e not in seen_edges:
                    seen_edges.add(e)
                    sg.edges.append(e)

        for c in sg.shown:
            sg.cn_full[c] = cn_full[c]
            direct = self._pair_edges(head, c, head, rel, gold, dd)
            sg.direct[c] = len(direct)
            if direct:
                add_pair(head, c)
            # наиболее специфичные посредники первыми (низкая степень ~ Adamic-Adar)
            xs = sorted(self.common(head, c, rel, gold, dd), key=lambda x: (self.degree(x), x))
            xs = xs[:cfg.max_paths_per_cand]
            sg.cn_shown[c] = len(xs)
            sg.paths[c] = xs
            for x in xs:
                # посредник, который сам показан кандидатом, остаётся кандидатом
                # (двойное объявление узла ломает раскладку dot); рёбра пути сохраняются
                if x not in shown_set:
                    inter_set.setdefault(x)
                add_pair(head, x)
                add_pair(x, c)
        sg.inter = list(inter_set)
        return sg


# ──────────────────────────────────────────────────────────────────
# Текстовые кодировки (из того же PathSubgraph, что и рендер)
# ──────────────────────────────────────────────────────────────────
def _label(sg: PathSubgraph, node: str, cfg: PathSubgraphConfig) -> str:
    if node in sg.inter and not cfg.label_intermediates:
        return f"x{sg.inter.index(node) + 1}"
    return node.replace("_", " ")


def encode_paths(sg: PathSubgraph, cfg: PathSubgraphConfig) -> str:
    """Список рёбер подграфа (как edge-list в основной статье)."""
    lines = [f"Candidates: {', '.join(_label(sg, c, cfg) for c in sg.shown)}"]
    for s, r, d in sg.edges:
        rel = f"--{r}-->" if cfg.label_edges else "-->"
        lines.append(f"- {_label(sg, s, cfg)} {rel} {_label(sg, d, cfg)}")
    return "\n".join(lines)


def encode_paths_budget(sg: PathSubgraph, cfg: PathSubgraphConfig,
                        count: Callable[[str], int], budget: int) -> Tuple[str, int, int]:
    """Edge-list того же подграфа, ужатый до budget токенов (iso-token с картинкой).

    Обрезка не «хвостом» (рёбра сгруппированы по кандидатам в алфавитном порядке —
    хвост выкинул бы последних кандидатов целиком), а по кругу: строка кандидатов
    целиком, затем прямые рёбра head–кандидат, затем 1-й путь каждого кандидата,
    2-й путь каждого… пока влезает. Посредники нумеруются в порядке появления.
    Возвращает (текст, показано путей, всего путей).
    """
    by_pair: Dict[frozenset, List[Tuple[str, str, str]]] = collections.defaultdict(list)
    for e in sg.edges:
        by_pair[frozenset((e[0], e[2]))].append(e)
    xid: Dict[str, str] = {}

    def lab(n: str) -> str:
        if n in sg.inter and not cfg.label_intermediates:
            return xid.setdefault(n, f"x{len(xid) + 1}")
        return n.replace("_", " ")

    def line(e: Tuple[str, str, str]) -> str:
        rel = f"--{e[1]}-->" if cfg.label_edges else "-->"
        return f"- {lab(e[0])} {rel} {lab(e[2])}"

    units: List[List[Tuple[str, str, str]]] = [by_pair[frozenset((sg.head, c))]
                                               for c in sg.shown if sg.direct.get(c)]
    total = sum(len(v) for v in sg.paths.values())
    for k in range(max((len(v) for v in sg.paths.values()), default=0)):
        for c in sg.shown:
            if k < len(sg.paths.get(c, ())):
                x = sg.paths[c][k]
                units.append(by_pair[frozenset((sg.head, x))] + by_pair[frozenset((x, c))])
    n_direct = sum(1 for c in sg.shown if sg.direct.get(c))

    out = [f"Candidates: {', '.join(lab(c) for c in sg.shown)}"]
    used = count(out[0])
    seen: Set[Tuple[str, str, str]] = set()
    kept_paths = 0
    for i, unit in enumerate(units):
        snap = dict(xid)
        new = [e for e in unit if e not in seen]
        lines = [line(e) for e in new]
        cost = sum(count("\n" + ln) for ln in lines)
        if used + cost > budget:
            xid.clear()
            xid.update(snap)        # откат нумерации посредников невлезшего пути
            break
        out.extend(lines)
        used += cost
        seen.update(new)
        kept_paths += i >= n_direct
    text = "\n".join(out)
    while len(out) > 1 and count(text) > budget:   # страховка: склейка ≠ сумма частей
        out.pop()
        text = "\n".join(out)
    return text, kept_paths, total


def encode_cn(sg: PathSubgraph, full: bool = True) -> str:
    """Предвычисленные структурные признаки: число 2-хоп путей head→candidate."""
    src = sg.cn_full if full else sg.cn_shown
    rows = [f"- {c.replace('_', ' ')}: {src[c]} shared neighbors"
            + (f", {sg.direct[c]} direct links" if sg.direct.get(c) else "")
            for c in sg.shown]
    return "Candidates (structural overlap with the head):\n" + "\n".join(rows)


# ──────────────────────────────────────────────────────────────────
# Рендер (стиль visualizer.FB15kGraphVisualizer)
# ──────────────────────────────────────────────────────────────────
def render(sg: PathSubgraph, cfg: PathSubgraphConfig, out_path: str,
           target_size: int = 560, render_dpi: int = 300, engine: str = "dot",
           max_label_len: int = 30, font_scale: float = 1.0) -> str:
    """Три слоя (LR): голова → посредники → кандидаты.

    Ребро рисуется в направлении слоя, а истинное направление передаётся
    стрелкой (dir=back), чтобы раскладка не ломала трёхслойную структуру.
    Если dot падает на ограничении rank=same («trouble in init_rank»), подграф
    перерисовывается без него — набор узлов и рёбер при этом тот же.
    """
    import graphviz
    base = out_path[:-4] if out_path.endswith(".png") else out_path
    try:
        _build_dot(sg, cfg, target_size, render_dpi, engine, max_label_len, True,
                   font_scale).render(base, cleanup=True)
    except graphviz.CalledProcessError:
        _build_dot(sg, cfg, target_size, render_dpi, engine, max_label_len, False,
                   font_scale).render(base, cleanup=True)
    path = base + ".png"
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    img.thumbnail((target_size, target_size))
    # прозрачные поля graphviz при convert("RGB") становятся чёрными → кладём на белый
    canvas = Image.new("RGB", (target_size, target_size), "white")
    canvas.paste(img, ((target_size - img.width) // 2, (target_size - img.height) // 2), img)
    canvas.save(path)
    return path


def _build_dot(sg: PathSubgraph, cfg: PathSubgraphConfig, target_size: int, render_dpi: int,
               engine: str, max_label_len: int, same_rank: bool, font_scale: float = 1.0):
    from graphviz import Digraph
    from visualizer import _nid, format_label

    size_in = f"{target_size / render_dpi:.2f}"
    dot = Digraph(format="png", engine=engine)
    dot.attr(dpi=str(render_dpi), size=f"{size_in},{size_in}!", ratio="fill",
             pad="0.1", margin="0.1", bgcolor="white")
    if engine == "dot":
        dot.attr(rankdir="LR", nodesep="0.15", ranksep="0.6", newrank="true")
    else:
        dot.attr(overlap="false", sep="+4", splines="true")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial Bold",
             fontsize=f"{13 * font_scale:g}", margin="0.12,0.06", height="0.35", width="0.6")
    dot.attr("edge", fontname="Arial", fontsize="11", fontcolor="gray25",
             penwidth="1.0", arrowsize="0.6")

    head = sg.head
    dot.node(_nid(head), format_label(head, max_label_len),
             fillcolor="lightskyblue", penwidth="2.5", fontsize=f"{14 * font_scale:g}")
    for x in sg.inter:
        if cfg.label_intermediates:
            dot.node(_nid(x), format_label(x, max_label_len), fillcolor="gray92",
                     fontsize="10")
        else:
            # shape=point не рисует подпись вовсе (пустой label graphviz заменяет именем узла)
            dot.node(_nid(x), shape="point", width="0.1", color="gray40")
    with dot.subgraph() as s:
        if same_rank:
            s.attr(rank="same")
        for c in sg.shown:
            s.node(_nid(c), format_label(c, max_label_len), fillcolor="lightyellow")

    layer = {head: 0, **{x: 1 for x in sg.inter}, **{c: 2 for c in sg.shown}}
    for s_, r, d in sg.edges:
        kw = {"label": f" {format_label(r, max_label_len)} "} if cfg.label_edges else {}
        if layer.get(s_, 1) <= layer.get(d, 1):
            dot.edge(_nid(s_), _nid(d), **kw)
        else:
            dot.edge(_nid(d), _nid(s_), dir="back", **kw)

    dot.attr(label=f"{format_label(head, max_label_len)} —[{format_label(sg.relation, max_label_len)}]→ ?",
             labelloc="t", fontsize="12", fontname="Arial Bold")
    return dot
