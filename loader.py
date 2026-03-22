from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase
from tqdm import tqdm

from config import LoaderConfig, Neo4jConfig
from utils import sanitize_relation_type, sanitize_text

logger = logging.getLogger(__name__)


class FB15k237Neo4jLoader:
    """Загружает FB15k-237 (train / valid / test) в Neo4j."""

    def __init__(self, cfg: LoaderConfig):
        self._cfg = cfg
        self._neo4j_cfg = cfg.neo4j
        self._driver = GraphDatabase.driver(
            self._neo4j_cfg.uri,
            auth=(self._neo4j_cfg.user, self._neo4j_cfg.password),
        )
        self.entity_names: Dict[str, str] = {}
        self.entity_descriptions: Dict[str, str] = {}

    # ── context manager ──────────────────────────────────────────
    def __enter__(self) -> "FB15k237Neo4jLoader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._driver.close()

    def _session(self):
        return self._driver.session(database=self._neo4j_cfg.database)

    # ── загрузка метаданных ──────────────────────────────────────
    def load_entity_names(self, filepath: str) -> int:
        count = 0
        logger.info("Загрузка имён сущностей из %s …", filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) >= 2:
                    self.entity_names[parts[0]] = sanitize_text(parts[1])
                    count += 1
        logger.info("Загружено %s имён сущностей", f"{count:,}")
        return count

    def load_entity_descriptions(self, filepath: str) -> int:
        if not os.path.exists(filepath):
            logger.warning("Файл описаний не найден: %s", filepath)
            return 0

        count = 0
        logger.info("Загрузка описаний из %s …", filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line == "null":
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2 or not parts[1].strip():
                    continue
                mid = parts[0]
                desc = re.sub(r'"@en$', "", parts[1])
                desc = re.sub(r'^"/m/[^\t]+', "", desc).strip()
                clean = sanitize_text(desc)
                if clean and mid in self.entity_names:
                    self.entity_descriptions[mid] = clean
                    count += 1
        logger.info("Загружено %s описаний", f"{count:,}")
        return count

    # ── индексы ──────────────────────────────────────────────────
    def _ensure_indexes(self, session) -> None:
        for q in (
            "CREATE INDEX entity_fb_id_index IF NOT EXISTS FOR (e:Entity) ON (e.fb_id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX relation_split_index IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.split)",
            "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.description]",
        ):
            session.run(q)
        logger.info("Индексы созданы / проверены")

    # ── загрузка троек ───────────────────────────────────────────
    def _get_display_name(self, mid: str) -> str:
        return self.entity_names.get(mid, mid.split("/")[-1].replace("_", " "))

    def load_triples(
        self,
        filepath: str,
        split: str,
        batch_size: int | None = None,
        remove_duplicates: bool | None = None,
    ) -> Tuple[int, int]:
        """Загружает тройки в Neo4j пакетно с меткой сплита."""
        batch_size = batch_size or self._cfg.batch_size
        remove_duplicates = (
            remove_duplicates if remove_duplicates is not None else self._cfg.remove_duplicates
        )

        logger.info("Загрузка троек из %s (split=%s) …", filepath, split)

        seen: Set[Tuple[str, str, str]] = set()
        triples: List[Tuple[str, str, str]] = []

        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                key = (parts[0], parts[1], parts[2])
                if remove_duplicates and key in seen:
                    continue
                seen.add(key)
                triples.append(key)

        logger.info("Уникальных троек: %s", f"{len(triples):,}")

        total_nodes = 0
        total_rels = 0

        with self._session() as session:
            self._ensure_indexes(session)
            for start in tqdm(range(0, len(triples), batch_size), desc=f"{split}"):
                batch = triples[start : start + batch_size]
                n, r = self._write_batch(session, batch, split)
                total_nodes += n
                total_rels += r

        return total_nodes, total_rels

    def _write_batch(
        self, session, batch: List[Tuple[str, str, str]], split: str
    ) -> Tuple[int, int]:
        items = []
        unique_entities: Set[str] = set()

        for head_mid, rel_raw, tail_mid in batch:
            h_name = self._get_display_name(head_mid)
            t_name = self._get_display_name(tail_mid)
            items.append(
                {
                    "head_fb_id": head_mid,
                    "head_name": h_name,
                    "head_desc": self.entity_descriptions.get(head_mid, h_name),
                    "tail_fb_id": tail_mid,
                    "tail_name": t_name,
                    "tail_desc": self.entity_descriptions.get(tail_mid, t_name),
                    "rel_type": sanitize_relation_type(rel_raw),
                    "split": split,
                }
            )
            unique_entities.update((head_mid, tail_mid))

        query = """
        UNWIND $items AS item
        MERGE (h:Entity {fb_id: item.head_fb_id})
          ON CREATE SET h.name = item.head_name,
                        h.description = item.head_desc,
                        h.created = timestamp()
          ON MATCH  SET h.name = COALESCE(h.name, item.head_name),
                        h.description = COALESCE(h.description, item.head_desc)
        MERGE (t:Entity {fb_id: item.tail_fb_id})
          ON CREATE SET t.name = item.tail_name,
                        t.description = item.tail_desc,
                        t.created = timestamp()
          ON MATCH  SET t.name = COALESCE(t.name, item.tail_name),
                        t.description = COALESCE(t.description, item.tail_desc)
        WITH h, t, item
        MERGE (h)-[r:RELATION {type: item.rel_type, split: item.split}]->(t)
          ON CREATE SET r.created = timestamp()
        RETURN count(r) AS rel_count
        """
        record = session.run(query, items=items).single()
        rel_count = record["rel_count"] if record else 0
        return len(unique_entities), rel_count

    # ── загрузка всех сплитов ────────────────────────────────────
    def load_all_splits(self) -> Dict[str, Tuple[int, int]]:
        """Загружает train / valid / test."""
        results: Dict[str, Tuple[int, int]] = {}
        for split_name, filename in (
            ("train", self._cfg.train_file),
            ("valid", self._cfg.valid_file),
            ("test", self._cfg.test_file),
        ):
            filepath = os.path.join(self._cfg.data_dir, filename)
            if not os.path.exists(filepath):
                logger.warning("Файл не найден: %s — пропуск split '%s'", filepath, split_name)
                continue
            n, r = self.load_triples(filepath, split_name)
            results[split_name] = (n, r)
            logger.info("%s: %s узлов, %s связей", split_name, f"{n:,}", f"{r:,}")
        return results

    # ── статистика ───────────────────────────────────────────────
    def get_stats(self, split: Optional[str] = None) -> Dict[str, Any]:
        with self._session() as session:
            entities = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]

            where = "WHERE r.split = $split" if split else ""
            params: Dict[str, Any] = {"split": split} if split else {}

            relations = session.run(
                f"MATCH ()-[r]->() {where} RETURN count(r) AS c", **params
            ).single()["c"]

            top_rels = [
                (r["type"], r["count"])
                for r in session.run(
                    f"MATCH ()-[r]->() {where} "
                    "RETURN type(r) AS type, count(*) AS count "
                    "ORDER BY count DESC LIMIT 10",
                    **params,
                )
            ]

            split_stats = {
                r["split"]: r["count"]
                for r in session.run(
                    "MATCH ()-[r]->() WHERE r.split IS NOT NULL "
                    "RETURN r.split AS split, count(r) AS count ORDER BY split"
                )
            }

        return {
            "entities": entities,
            "relations": relations,
            "top_relations": top_rels,
            "split_stats": split_stats,
        }