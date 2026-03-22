from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
from neo4j import GraphDatabase

from config import EnricherConfig

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError("pip install sentence-transformers") from exc

logger = logging.getLogger(__name__)


class SemanticEnricher:
    """
    Трёхэтапное обогащение:
      1) embed(node_text) → e.embedding
      2) cosine(h, t) → r.semantic_weight  (без self-loops)
      3) TransE: 1/(1+‖h+r−t‖) → r.transe_score
    """

    def __init__(self, cfg: EnricherConfig):
        self._cfg = cfg
        self._driver = GraphDatabase.driver(
            cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
        )

        logger.info("Загрузка модели: %s", cfg.embedding_model)
        t0 = time.time()
        self._model = SentenceTransformer(
            cfg.embedding_model, device=cfg.device, trust_remote_code=True,
        )
        self._dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Модель загружена за %.1fs, dim=%d", time.time() - t0, self._dim,
        )

    def __enter__(self) -> "SemanticEnricher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    def _session(self):
        return self._driver.session(database=self._cfg.neo4j.database)

    # ══════════════════════════════════════════════════════════════
    # Этап 1: эмбеддинги узлов
    # ══════════════════════════════════════════════════════════════

    def _embed_nodes(self) -> int:
        batch_size = self._cfg.node_batch_size
        desc_max = self._cfg.description_max_len

        logger.info("Этап 1/3: эмбеддинги узлов …")

        with self._session() as s:
            records = list(s.run(
                "MATCH (e:Entity) WHERE e.embedding IS NULL "
                "RETURN e.fb_id AS fid, e.name AS name, e.description AS desc",
            ))

        if not records:
            logger.info("Все узлы уже имеют эмбеддинги")
            return 0

        n = len(records)
        logger.info("Узлов: %s", f"{n:,}")

        fb_ids: List[str] = []
        texts: List[str] = []
        for r in records:
            name = (r["name"] or "").strip()
            desc = (r["desc"] or "").strip()[:desc_max]
            texts.append(f"{name}. {desc}" if desc else (name or "unknown"))
            fb_ids.append(r["fid"])

        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True,
            batch_size=batch_size, convert_to_numpy=True,
        )

        updates = [
            {"fid": fid, "emb": emb.tolist()}
            for fid, emb in zip(fb_ids, embeddings)
        ]
        with self._session() as s:
            s.run(
                "UNWIND $updates AS u "
                "MATCH (e:Entity {fb_id: u.fid}) "
                "SET e.embedding = u.emb, "
                "    e.embedding_model = $model, "
                "    e.embedding_dim = $dim, "
                "    e.embedded_at = timestamp()",
                updates=updates, model=self._cfg.embedding_model, dim=self._dim,
            )

        self._ensure_vector_index()
        logger.info("Этап 1/3 завершён: %s узлов", f"{n:,}")
        return n

    def _ensure_vector_index(self) -> None:
        with self._session() as s:
            exists = s.run(
                "SHOW INDEXES YIELD name "
                "WHERE name = 'entity_embedding_index' "
                "RETURN count(*) AS c",
            ).single()["c"]
            if exists:
                return
            s.run(
                "CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS "
                "FOR (e:Entity) ON e.embedding "
                f"OPTIONS {{indexConfig: {{"
                f"  `vector.dimensions`: {self._dim},"
                f"  `vector.similarity_function`: 'cosine'"
                f"}}}}",
            )

    # ══════════════════════════════════════════════════════════════
    # Этап 2: cosine similarity (без self-loops)
    # ══════════════════════════════════════════════════════════════

    def _compute_cosine_weights(self) -> int:
        batch_size = self._cfg.edge_batch_size

        logger.info("Этап 2/3: cosine similarity …")

        # Исключаем self-loops (h.name = t.name) — они всегда дают 1.0
        with self._session() as s:
            records = list(s.run(
                "MATCH (h:Entity)-[r:RELATION]->(t:Entity) "
                "WHERE r.semantic_weight IS NULL "
                "  AND h.embedding IS NOT NULL "
                "  AND t.embedding IS NOT NULL "
                "  AND h.name <> t.name "
                "RETURN "
                "  elementId(r) AS rid, "
                "  h.embedding AS h_emb, "
                "  t.embedding AS t_emb, "
                "  h.name AS h_name, "
                "  r.type AS rel_type, "
                "  t.name AS t_name",
            ))

        # Отдельно обрабатываем self-loops: ставим 0.0 (не информативны)
        with self._session() as s:
            self_loop_count = s.run(
                "MATCH (h:Entity)-[r:RELATION]->(t:Entity) "
                "WHERE r.semantic_weight IS NULL "
                "  AND h.name = t.name "
                "SET r.semantic_weight = 0.0 "
                "RETURN count(r) AS c",
            ).single()["c"]
        if self_loop_count:
            logger.info("Self-loops: %s рёбер → weight=0.0", f"{self_loop_count:,}")

        if not records:
            logger.info("Все не-self-loop рёбра уже имеют semantic_weight")
            return self_loop_count

        # Фильтрация по размерности
        valid = [
            r for r in records
            if (isinstance(r["h_emb"], (list, tuple))
                and isinstance(r["t_emb"], (list, tuple))
                and len(r["h_emb"]) == self._dim
                and len(r["t_emb"]) == self._dim)
        ]
        skipped = len(records) - len(valid)
        if skipped:
            logger.warning("Пропущено %s рёбер (dim ≠ %d)", f"{skipped:,}", self._dim)
        if not valid:
            return self_loop_count

        n = len(valid)
        logger.info("Рёбер для cosine: %s", f"{n:,}")

        h_embs = np.array([r["h_emb"] for r in valid], dtype=np.float32)
        t_embs = np.array([r["t_emb"] for r in valid], dtype=np.float32)
        sims = np.clip(np.einsum("ij,ij->i", h_embs, t_embs), 0.0, 1.0)

        logger.info(
            "Cosine (без self-loops): mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            sims.mean(), sims.std(), sims.min(), sims.max(),
        )

        self._save_edge_property(valid, sims, "semantic_weight", batch_size)
        logger.info("Этап 2/3 завершён: %s рёбер", f"{n + self_loop_count:,}")
        return n + self_loop_count

    # ══════════════════════════════════════════════════════════════
    # Этап 3: TransE-style scoring
    # ══════════════════════════════════════════════════════════════

    def _compute_transe_scores(self) -> int:
        """score = 1 / (1 + ‖h + r − t‖₂)"""
        batch_size = self._cfg.edge_batch_size

        logger.info("Этап 3/3: TransE scoring …")

        with self._session() as s:
            records = list(s.run(
                "MATCH (h:Entity)-[r:RELATION]->(t:Entity) "
                "WHERE r.transe_score IS NULL "
                "  AND h.embedding IS NOT NULL "
                "  AND t.embedding IS NOT NULL "
                "RETURN "
                "  elementId(r) AS rid, "
                "  h.embedding AS h_emb, "
                "  t.embedding AS t_emb, "
                "  r.type AS rel_type, "
                "  h.name AS h_name, "
                "  t.name AS t_name",
            ))

        if not records:
            logger.info("Все рёбра уже имеют transe_score")
            return 0

        valid = [
            r for r in records
            if (isinstance(r["h_emb"], (list, tuple))
                and isinstance(r["t_emb"], (list, tuple))
                and len(r["h_emb"]) == self._dim
                and len(r["t_emb"]) == self._dim)
        ]
        if not valid:
            logger.error("Нет валидных рёбер для TransE")
            return 0

        n = len(valid)
        logger.info("Рёбер: %s", f"{n:,}")

        # Эмбеддинги типов отношений
        unique_rels = sorted({r["rel_type"] for r in valid})
        logger.info("Уникальных relation types: %d", len(unique_rels))

        rel_texts = [rel.replace("_", " ") for rel in unique_rels]
        rel_embeddings = self._model.encode(
            rel_texts, normalize_embeddings=True,
            batch_size=min(512, len(rel_texts)),
            convert_to_numpy=True,
        )
        rel_emb_map = dict(zip(unique_rels, rel_embeddings))

        # Vectorized TransE
        h_embs = np.array([r["h_emb"] for r in valid], dtype=np.float32)
        t_embs = np.array([r["t_emb"] for r in valid], dtype=np.float32)
        r_embs = np.array(
            [rel_emb_map[r["rel_type"]] for r in valid], dtype=np.float32,
        )

        # ‖h + r − t‖₂
        distances = np.linalg.norm(h_embs + r_embs - t_embs, axis=1)

        # score ∈ (0, 1]
        scores = 1.0 / (1.0 + distances)

        # Self-loops: head + rel ≈ head → distance маленький → score высокий
        # Это корректно для TransE: self-loop означает rel ≈ 0, h ≈ t
        # Но можно занулить если нужно:
        for i, r in enumerate(valid):
            if r["h_name"] == r["t_name"]:
                scores[i] = 0.0

        logger.info(
            "TransE scores: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            scores.mean(), scores.std(), scores.min(), scores.max(),
        )

        # Примеры
        sorted_idx = np.argsort(scores)[::-1]
        logger.info("Топ-5 TransE:")
        for i in sorted_idx[:5]:
            logger.info(
                "  (%s) + [%s] − (%s) = dist=%.3f, score=%.4f",
                valid[i]["h_name"], valid[i]["rel_type"],
                valid[i]["t_name"], distances[i], scores[i],
            )
        logger.info("Bottom-5 TransE:")
        for i in sorted_idx[-5:]:
            logger.info(
                "  (%s) + [%s] − (%s) = dist=%.3f, score=%.4f",
                valid[i]["h_name"], valid[i]["rel_type"],
                valid[i]["t_name"], distances[i], scores[i],
            )

        self._save_edge_property(valid, scores, "transe_score", batch_size)
        logger.info("Этап 3/3 завершён: %s рёбер", f"{n:,}")
        return n

    # ══════════════════════════════════════════════════════════════
    # Сохранение
    # ══════════════════════════════════════════════════════════════

    def _save_edge_property(
        self,
        records: List[Dict],
        values: np.ndarray,
        property_name: str,
        batch_size: int,
    ) -> None:
        """Батчевое сохранение float-свойства на рёбрах."""
        n = len(records)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            updates = [
                {"rid": records[i]["rid"], "val": float(values[i])}
                for i in range(start, end)
            ]
            with self._session() as s:
                s.run(
                    "UNWIND $updates AS u "
                    "MATCH ()-[r]-() "
                    "WHERE elementId(r) = u.rid "
                    f"SET r.{property_name} = u.val, "
                    f"    r.{property_name}_at = timestamp()",
                    updates=updates,
                )

    # ══════════════════════════════════════════════════════════════
    # Публичный API
    # ══════════════════════════════════════════════════════════════

    def enrich_all(self) -> Dict[str, Any]:
        """1) node embeddings → 2) cosine weights → 3) TransE scores"""
        t0 = time.time()

        nodes = self._embed_nodes()
        cosine = self._compute_cosine_weights()
        transe = self._compute_transe_scores()

        elapsed = time.time() - t0
        result = {
            "nodes_embedded": nodes,
            "edges_cosine": cosine,
            "edges_transe": transe,
            "total_time_sec": round(elapsed, 1),
        }
        logger.info("Обогащение: %.1fs, %s", elapsed, result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        with self._session() as s:
            total_nodes = s.run(
                "MATCH (e:Entity) RETURN count(e) AS c"
            ).single()["c"]
            embedded = s.run(
                "MATCH (e:Entity) WHERE e.embedding IS NOT NULL "
                "RETURN count(e) AS c"
            ).single()["c"]
            total_edges = s.run(
                "MATCH ()-[r:RELATION]->() RETURN count(r) AS c"
            ).single()["c"]
            cosine_edges = s.run(
                "MATCH ()-[r:RELATION]->() "
                "WHERE r.semantic_weight IS NOT NULL "
                "RETURN count(r) AS c"
            ).single()["c"]
            transe_edges = s.run(
                "MATCH ()-[r:RELATION]->() "
                "WHERE r.transe_score IS NOT NULL "
                "RETURN count(r) AS c"
            ).single()["c"]
            self_loops = s.run(
                "MATCH (h:Entity)-[r:RELATION]->(t:Entity) "
                "WHERE h.name = t.name "
                "RETURN count(r) AS c"
            ).single()["c"]

        return {
            "nodes_total": total_nodes,
            "nodes_embedded": embedded,
            "edges_total": total_edges,
            "edges_cosine": cosine_edges,
            "edges_transe": transe_edges,
            "self_loops": self_loops,
        }