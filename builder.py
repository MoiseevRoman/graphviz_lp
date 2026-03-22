from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict
from joblib import Parallel, delayed
from tqdm import tqdm

from config import BuilderConfig
from generators import Stage1DataGenerator, Stage2DataGenerator
from retriever import FB15k237GraphRetriever
from visualizer import FB15kGraphVisualizer, visualize_task

logger = logging.getLogger(__name__)


class FB15k237GraphVisBuilder:

    def __init__(self, cfg: BuilderConfig):
        self._cfg = cfg
        random.seed(cfg.seed)
        self._retriever = FB15k237GraphRetriever(cfg.retriever)
        self._vis = FB15kGraphVisualizer(cfg.visualization)
        self._stage1 = Stage1DataGenerator()
        self._stage2 = Stage2DataGenerator()
        self._entity_names: Dict[str, str] = {}
        self.stats: Dict[str, int] = {
            "processed": 0, "errors": 0,
            "stage1_examples": 0, "stage2_examples": 0,
        }

    def __enter__(self) -> "FB15k237GraphVisBuilder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._retriever.close()

    # ── загрузка имён / троек ────────────────────────────────────

    def _load_entity_names(self) -> None:
        filepath = os.path.join(self._cfg.data_dir, self._cfg.entity_names_file)
        if not os.path.exists(filepath):
            logger.warning("Файл имён не найден: %s", filepath)
            return
        count = 0
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t", 1)
                if len(parts) >= 2:
                    self._entity_names[parts[0]] = parts[1].strip()
                    count += 1
        logger.info("Загружено %s имён", f"{count:,}")

    def _mid_to_name(self, mid: str) -> str:
        return self._entity_names.get(
            mid, mid.split("/")[-1].replace("_", " "),
        )

    def _load_triples_from_file(
        self, filepath: str, limit: Optional[int] = None,
    ) -> List[Tuple[str, str, str]]:
        if not os.path.exists(filepath):
            logger.warning("Файл не найден: %s", filepath)
            return []
        triples: List[Tuple[str, str, str]] = []
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                h_mid, rel_raw, t_mid = parts
                rel = rel_raw.split("/")[-1] or rel_raw
                triples.append((
                    self._mid_to_name(h_mid), rel, self._mid_to_name(t_mid),
                ))
                if limit and len(triples) >= limit:
                    break
        logger.info("Загружено %s троек из %s", f"{len(triples):,}", filepath)
        return triples

    def _load_all_splits(
        self, limit: Optional[int] = None,
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        d = self._cfg.data_dir
        return {
            "train": self._load_triples_from_file(
                os.path.join(d, self._cfg.train_file), limit,
            ),
            "valid": self._load_triples_from_file(
                os.path.join(d, self._cfg.valid_file), limit,
            ),
            "test": self._load_triples_from_file(
                os.path.join(d, self._cfg.test_file), limit,
            ),
        }

    # ── вспомогательные ──────────────────────────────────────────

    @staticmethod
    def _query_id(head: str, rel: str, tail: str) -> str:
        return hashlib.md5(
            f"{head}:{rel}:{tail}".encode(),
        ).hexdigest()[:16]

    def _random_splits(self, triples: List[Tuple]) -> Dict[str, List[Tuple]]:
        random.seed(self._cfg.seed)
        shuffled = list(triples)
        random.shuffle(shuffled)
        n = len(shuffled)
        t = int(n * self._cfg.train_ratio)
        v = int(n * (self._cfg.train_ratio + self._cfg.valid_ratio))
        return {"train": shuffled[:t], "valid": shuffled[t:v], "test": shuffled[v:]}

    # ── параллельная визуализация ────────────────────────────────

    def _visualize_parallel(
        self,
        tasks: List[Tuple[Dict, str, Optional[str]]],
        num_workers: int,
    ) -> List[str]:
        vis_cfg = self._cfg.visualization
        return Parallel(n_jobs=num_workers, backend="loky")(
            delayed(visualize_task)(sg, qid, trel, vis_cfg)
            for sg, qid, trel in tasks
        )

    # ── извлечение подграфов ─────────────────────────────────────

    def _extract_subgraphs(
        self,
        triples: List[Tuple[str, str, str]],
        split: str,
    ) -> Tuple[
        List[Tuple[Dict, str, Optional[str]]],
        List[Tuple[str, str, str]],
    ]:
        """Извлекает HEAD-centric подграфы.

        Returns:
            tasks: [(subgraph_copy, query_id, relation), ...]
            valid_triples: [(head, rel, tail), ...] — tail только как label
        """
        tasks: List[Tuple[Dict, str, Optional[str]]] = []
        valid_triples: List[Tuple[str, str, str]] = []

        for head, rel, tail in tqdm(triples, desc=f"Подграфы ({split})"):
            try:
                qid = self._query_id(head, rel, tail)

                sg = self._retriever.get_head_subgraph(
                    head=head,
                    target_rel=rel,
                    target_tail=tail,
                    split=split,
                )

                if "error" in sg or not sg.get("edges"):
                    # Fallback: 1-hop тоже исключает tail
                    sg = self._retriever.get_1hop_neighbors(
                        head, exclude_entity=tail,
                    )
                    if not sg.get("edges"):
                        self.stats["errors"] += 1
                        continue
                    sg["rel"] = rel

                sg_copy: Dict[str, Any] = {
                    "head": sg["head"],
                    "rel": sg["rel"],
                    "nodes": list(sg.get("nodes", [])),
                    "edges": list(sg.get("edges", [])),
                    "stats": dict(sg.get("stats", {})),
                    "head_description": sg.get("head_description", ""),
                }

                tasks.append((sg_copy, qid, rel))
                valid_triples.append((head, rel, tail))

            except Exception:
                self.stats["errors"] += 1
                logger.debug(
                    "Ошибка подграфа (%s, %s, %s)",
                    head, rel, tail, exc_info=True,
                )

        return tasks, valid_triples

    # ── Stage 1 ──────────────────────────────────────────────────

    def build_stage1(
        self,
        triples: List[Tuple[str, str, str]],
        output_path: str,
        num_workers: int | None = None,
    ) -> Dataset:
        workers = num_workers or self._cfg.num_visualization_workers
        logger.info("Stage 1: %d примеров, workers=%d", len(triples), workers)

        tasks, _ = self._extract_subgraphs(triples, split="stage1")
        if not tasks:
            return Dataset.from_list([])

        image_paths = self._visualize_parallel(tasks, workers)

        data: List[Dict[str, Any]] = []
        for (sg, qid, _), img_path in zip(tasks, image_paths):
            data.extend(self._stage1.generate(sg, img_path, qid))
        self.stats["stage1_examples"] += len(data)

        ds = Dataset.from_list(data)
        ds.save_to_disk(output_path)
        logger.info("Stage 1: %s (%d примеров)", output_path, len(data))
        return ds

    # ── Stage 2 ──────────────────────────────────────────────────

    def build_stage2(
        self,
        triples: List[Tuple[str, str, str]],
        split: str,
        output_path: str,
        num_workers: int | None = None,
    ) -> Dataset:
        workers = num_workers or self._cfg.num_visualization_workers
        logger.info(
            "Stage 2 (%s): %d примеров, workers=%d",
            split, len(triples), workers,
        )

        tasks, valid_triples = self._extract_subgraphs(triples, split=split)
        if not tasks:
            return Dataset.from_list([])

        image_paths = self._visualize_parallel(tasks, workers)

        data: List[Dict[str, Any]] = []
        for (sg, qid, _), img_path, (_, _, tail) in zip(
            tasks, image_paths, valid_triples,
        ):
            try:
                example = self._stage2.generate(
                    head=sg["head"],
                    relation=sg.get("rel", ""),
                    tail=tail,
                    head_description=sg.get("head_description", ""),
                    image_path=img_path,
                    query_id=qid,
                )
                data.append(example)
            except Exception:
                self.stats["errors"] += 1
                logger.debug("Ошибка Stage 2", exc_info=True)

        self.stats["stage2_examples"] += len(data)
        ds = Dataset.from_list(data)
        ds.save_to_disk(output_path)
        logger.info("Stage 2 %s: %s (%d)", split, output_path, len(data))
        return ds

    # ── полный пайплайн ──────────────────────────────────────────

    def build(self, num_workers: int | None = None) -> DatasetDict:
        workers = num_workers or self._cfg.num_visualization_workers
        output_base = Path(self._cfg.visualization.output_dir) / "datasets"
        output_base.mkdir(parents=True, exist_ok=True)

        self._load_entity_names()
        limit = self._cfg.max_triples_for_dataset

        if self._cfg.use_split_property:
            splits = self._load_all_splits(limit)
            train_t, valid_t, test_t = (
                splits["train"], splits["valid"], splits["test"],
            )
        else:
            all_t = self._load_triples_from_file(
                os.path.join(self._cfg.data_dir, self._cfg.train_file), limit,
            )
            s = self._random_splits(all_t)
            train_t, valid_t, test_t = s["train"], s["valid"], s["test"]

        logger.info(
            "train=%d, valid=%d, test=%d",
            len(train_t), len(valid_t), len(test_t),
        )

        s1_sample = (
            random.sample(
                train_t,
                min(self._cfg.sample_size_stage1, len(train_t)),
            )
            if self._cfg.sample_size_stage1 else train_t
        )
        s1 = self.build_stage1(
            s1_sample,
            str(output_base / "stage1_graph_comprehension"),
            workers,
        )
        s2_train = self.build_stage2(
            train_t, "train", str(output_base / "stage2_train"), workers,
        )
        s2_valid = self.build_stage2(
            valid_t, "valid", str(output_base / "stage2_valid"), workers,
        )
        s2_test = self.build_stage2(
            test_t, "test", str(output_base / "stage2_test"), workers,
        )

        dd = DatasetDict({
            "stage1": s1, "stage2_train": s2_train,
            "stage2_valid": s2_valid, "stage2_test": s2_test,
        })
        dd.save_to_disk(str(output_base))

        meta = {
            "stats": self.stats,
            "created_at": datetime.now().isoformat(),
            "splits": {
                "train": len(train_t),
                "valid": len(valid_t),
                "test": len(test_t),
            },
        }
        (output_base / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info("Датасет: %s | %s", output_base, self.stats)
        return dd