#!/usr/bin/env python3
"""Быстрая проверка доступности Neo4j перед дорогой сборкой датасета.

Использует те же креды, что rerender_dataset.py: ``Neo4jConfig.from_env()``
(env NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE; дефолт
bolt://localhost:7687, neo4j / password, db=neo4j).

Exit 0 = БД отвечает; exit 1 = недоступна (с понятным сообщением). Так пайплайн
падает сразу и осмысленно, а не 60-строчным трейсбеком внутри рендера.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as _cfg  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


def main() -> int:
    c = _cfg.Neo4jConfig.from_env()
    try:
        drv = GraphDatabase.driver(c.uri, auth=(c.user, c.password))
        drv.verify_connectivity()
        with drv.session(database=c.database) as s:
            n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        drv.close()
        print(f"[neo4j] OK: {c.uri} db={c.database} nodes={n}")
        if n == 0:
            print("[neo4j] ВНИМАНИЕ: граф пуст (0 узлов) — сначала загрузите FB15k-237 "
                  "(load_graph.py + enrich.py).")
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[neo4j] НЕДОСТУПЕН: {c.uri}  ({exc.__class__.__name__})")
        print("        Запустите Neo4j с загруженным графом FB15k-237 и повторите.")
        print("        Креды берутся из env NEO4J_URI/USER/PASSWORD (дефолт neo4j/password).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
