"""CLI: загрузка FB15k-237 в Neo4j.

Usage:
    python -m graphvis_lp.scripts.load_graph
"""

from __future__ import annotations

import logging

from graphvis_lp.config import LoaderConfig, Neo4jConfig
from graphvis_lp.loader import FB15k237Neo4jLoader


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger(__name__)

    cfg = LoaderConfig(neo4j=Neo4jConfig.from_env())

    with FB15k237Neo4jLoader(cfg) as loader:
        import os

        loader.load_entity_names(os.path.join(cfg.data_dir, cfg.entity_names_file))
        loader.load_entity_descriptions(os.path.join(cfg.data_dir, cfg.entity_desc_file))

        results = loader.load_all_splits()

        stats = loader.get_stats()
        log.info("Entities: %s", f"{stats['entities']:,}")
        log.info("Relations: %s", f"{stats['relations']:,}")
        for split_name, count in stats["split_stats"].items():
            log.info("  %s: %s", split_name, f"{count:,}")


if __name__ == "__main__":
    main()