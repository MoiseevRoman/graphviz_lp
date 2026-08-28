"""CLI: семантическое обогащение графа.

Usage:
    python -m graphvis_lp.scripts.enrich_graph
"""

from __future__ import annotations

import logging

from config import EnricherConfig, Neo4jConfig
from enricher import SemanticEnricher


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

cfg = EnricherConfig(neo4j=Neo4jConfig.from_env())

with SemanticEnricher(cfg) as enricher:
    result = enricher.enrich_all()
    logging.getLogger(__name__).info("Результат: %s", result)
