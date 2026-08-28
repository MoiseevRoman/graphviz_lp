"""CLI: сборка визуального датасета.

Usage:
    python -m graphvis_lp.scripts.build_dataset
"""

from __future__ import annotations

import logging

from graphvis_lp.builder import FB15k237GraphVisBuilder
from graphvis_lp.config import (
    BuilderConfig,
    Neo4jConfig,
    RetrieverConfig,
    VisualizationConfig,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    neo4j_cfg = Neo4jConfig.from_env()
    cfg = BuilderConfig(
        neo4j=neo4j_cfg,
        retriever=RetrieverConfig(neo4j=neo4j_cfg),
        visualization=VisualizationConfig(),
    )

    with FB15k237GraphVisBuilder(cfg) as builder:
        builder.build()


if __name__ == "__main__":
    main()