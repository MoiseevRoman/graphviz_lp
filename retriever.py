from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase

from config import RetrieverConfig

logger = logging.getLogger(__name__)

_SPLIT_FILTERS = {
    "train": ["train"],
    "valid": ["train", "valid"],
    "test": ["train", "valid", "test"],
    "stage1": ["train"],
}


class FB15k237GraphRetriever:
    """Извлечение head-centric подграфов. Top-K соседей по semantic_weight."""

    def __init__(self, cfg: RetrieverConfig):
        self._cfg = cfg
        self._neo4j_cfg = cfg.neo4j
        self._driver = GraphDatabase.driver(
            self._neo4j_cfg.uri,
            auth=(self._neo4j_cfg.user, self._neo4j_cfg.password),
        )
        self._ensure_indexes()
        self._weights_available: Optional[bool] = None

    def __enter__(self) -> "FB15k237GraphRetriever":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    @property
    def driver(self):
        return self._driver

    def _session(self):
        return self._driver.session(database=self._neo4j_cfg.database)

    def _ensure_indexes(self) -> None:
        try:
            with self._session() as s:
                s.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                s.run("CREATE INDEX entity_fb_id_index IF NOT EXISTS FOR (e:Entity) ON (e.fb_id)")
        except Exception as exc:
            logger.warning("Предупреждение при создании индексов: %s", exc)

    def _check_weights_exist(self) -> bool:
        if self._weights_available is not None:
            return self._weights_available
        prop = self._cfg.weight_property
        with self._session() as s:
            count = s.run(
                f"MATCH ()-[r:RELATION]->() "
                f"WHERE r.{prop} IS NOT NULL "
                f"RETURN count(r) AS c LIMIT 1",
            ).single()["c"]
        self._weights_available = count > 0
        if not self._weights_available:
            logger.warning(
                "%s не найден. Запустите: python -m graphvis_lp.scripts.enrich_graph",
                prop,
            )
        return self._weights_available

    @staticmethod
    def _empty_subgraph(error: str = "entity_not_found") -> Dict[str, Any]:
        return {
            "head": "", "rel": "",
            "nodes": [], "edges": [],
            "head_description": "", "error": error,
            "stats": {"total_nodes": 0, "total_edges": 0},
        }

    @staticmethod
    def _edge_key(src: str, tgt: str, rel: str) -> Tuple[str, ...]:
        return (*sorted((src, tgt)), rel)

    def _allowed_splits(self, split: str) -> List[str]:
        return _SPLIT_FILTERS.get(split, ["train", "valid", "test"])

    def _order_clause(self) -> str:
        prop = self._cfg.weight_property
        if self._cfg.use_semantic_weight and self._check_weights_exist():
            return f"ORDER BY COALESCE(r.{prop}, 0.0) DESC"
        return ""

    # ── основной метод ────────────────────────────────────────────

    def get_head_subgraph(
        self,
        head: str,
        target_rel: str,
        target_tail: str,
        split: str = "train",
    ) -> Dict[str, Any]:
        """Извлекает head-centric подграф. Top-K соседей по весу.

        Tail полностью исключён из узлов и рёбер.
        """
        k = self._cfg.k_neighbors
        allowed_splits = self._allowed_splits(split)
        order = self._order_clause()

        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, str]] = []
        seen_edges: Set[Tuple[str, ...]] = set()

        def _add_edge(src: str, tgt: str, relation: str) -> bool:
            if src == target_tail or tgt == target_tail:
                return False
            if self._cfg.remove_duplicate_edges:
                key = self._edge_key(src, tgt, relation)
                if key in seen_edges:
                    return False
                seen_edges.add(key)
            edges.append({"source": src, "target": tgt, "relation": relation})
            return True

        with self._session() as session:
            # 1) Head
            rec = session.run(
                "MATCH (e:Entity {name: $name}) "
                "RETURN e.fb_id AS fid, e.name AS name, "
                "  e.description AS desc",
                name=head,
            ).single()
            if rec is None:
                return self._empty_subgraph()

            head_desc = (rec["desc"] or "")[:200]
            nodes[head] = {
                "fb_id": rec["fid"],
                "name": rec["name"],
                "description": head_desc,
            }

            # 2) Top-K 1-hop соседей (без tail)
            query = (
                "MATCH (h:Entity {name: $head})-[r]-(nb:Entity) "
                "WHERE r.split IN $splits "
                "  AND nb.name <> $tail "
                "RETURN "
                "  nb.name AS name, nb.fb_id AS fid, "
                "  nb.description AS desc, "
                "  r.type AS rel, "
                "  COALESCE(r.semantic_weight, 0.0) AS w "
                f"{order} "
                "LIMIT $k"
            )
            for r in session.run(
                query, head=head, splits=allowed_splits,
                tail=target_tail, k=k,
            ):
                nb_name = r["name"]
                if nb_name not in nodes:
                    nodes[nb_name] = {
                        "fb_id": r["fid"],
                        "name": nb_name,
                        "description": (r["desc"] or "")[:80],
                    }
                _add_edge(head, nb_name, r["rel"])

        return {
            "head": head,
            "rel": target_rel,
            "nodes": list(nodes.values()),
            "edges": edges,
            "head_description": head_desc,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }

    # ── fallback ──────────────────────────────────────────────────

    def get_1hop_neighbors(
        self,
        entity: str,
        exclude_entity: str = "",
        max_neighbors: int | None = None,
    ) -> Dict[str, Any]:
        """1-hop с ранжированием по весу."""
        k = max_neighbors or self._cfg.k_neighbors
        order = self._order_clause()

        nodes: Dict[str, Dict] = {}
        edges: List[Dict] = []

        with self._session() as session:
            rec = session.run(
                "MATCH (e:Entity {name: $name}) "
                "RETURN e.fb_id AS fid, e.name AS name, "
                "  e.description AS desc",
                name=entity,
            ).single()
            if rec is None:
                return self._empty_subgraph()

            head_desc = (rec["desc"] or "")[:200]
            nodes[entity] = {
                "fb_id": rec["fid"],
                "name": rec["name"],
                "description": head_desc,
            }

            query = (
                "MATCH (a:Entity {name: $name})-[r]-(nb:Entity) "
                "WHERE nb.name <> $exclude "
                "RETURN nb.name AS nn, nb.fb_id AS nid, "
                "  nb.description AS nd, r.type AS rel, "
                "  COALESCE(r.semantic_weight, 0.0) AS w "
                f"{order} "
                "LIMIT $k"
            )
            for r in session.run(
                query, name=entity, exclude=exclude_entity, k=k,
            ):
                nb = r["nn"]
                if nb not in nodes:
                    nodes[nb] = {
                        "fb_id": r["nid"],
                        "name": nb,
                        "description": (r["nd"] or "")[:80],
                    }
                edges.append({
                    "source": entity,
                    "target": nb,
                    "relation": r["rel"],
                })

        return {
            "head": entity,
            "rel": "",
            "nodes": list(nodes.values()),
            "edges": edges,
            "head_description": head_desc,
            "stats": {"total_nodes": len(nodes), "total_edges": len(edges)},
        }