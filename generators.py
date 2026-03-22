"""Генераторы Stage 1 и Stage 2. Leak-free: tail нигде не фигурирует."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List

# Имя узла-заглушки — ровно то, что нарисовано на изображении
_QUESTION_NODE = "[?]"


# ════════════════════════════════════════════════════════════════
# Stage 1 — Visual Graph Comprehension
# ════════════════════════════════════════════════════════════════

class Stage1DataGenerator:
    """QA по тому, что модель ВИДИТ на изображении.

    На изображении присутствуют:
      - все узлы из subgraph["nodes"]
      - узел-заглушка [?]
      - все рёбра из subgraph["edges"]
      - пунктирное ребро head —[rel]→ [?]

    Поэтому ответы используют «визуальные» подсчёты:
      visible_nodes = real_nodes + 1  ([?])
      visible_edges = real_edges + 1  (head→[?])
      degree(head) += 1               (ребро к [?])
      degree([?])  = 1
    """

    _PROMPTS: Dict[str, List[str]] = {
        "node_count": [
            "How many nodes are there in the graph?",
            "Count the total number of nodes shown.",
            "What is the total number of vertices in this graph?",
            "How many entities are depicted in the image?",
            "Tell me the node count of this knowledge graph.",
        ],
        "edge_count": [
            "How many edges are there in the graph?",
            "Count the total number of connections.",
            "What is the total number of edges in this graph?",
            "How many relationships are shown in the image?",
            "Tell me the edge count of this knowledge graph.",
        ],
        "node_degree": [
            "What is the degree of the node '{node}'?",
            "How many connections does '{node}' have?",
            "What is the number of edges incident to '{node}'?",
            "Tell me the degree of vertex '{node}'.",
            "How many neighbors does the node '{node}' have?",
        ],
        "highest_degree": [
            "Which node has the highest degree in the graph?",
            "Identify the node with the most connections.",
            "What is the name of the node with maximum degree?",
            "Which entity has the most relationships in this graph?",
            "Find the node with the highest number of edges.",
        ],
        "node_listing": [
            "List all nodes shown in the graph.",
            "Name all entities depicted in the image.",
            "What are all the nodes in this knowledge graph?",
            "Enumerate all vertices shown.",
            "Provide a list of all node names in the graph.",
        ],
        "triple_listing": [
            "List all triples in the graph.",
            "Enumerate all (head, relation, tail) triples shown.",
            "What are all the relationships in this graph?",
            "List all edges as (source, label, target) triples.",
            "Describe the graph by listing all its triples.",
        ],
    }

    @staticmethod
    def _compute_degrees(edges: List[Dict[str, str]]) -> Counter:
        deg: Counter = Counter()
        for e in edges:
            deg[e["source"]] += 1
            deg[e["target"]] += 1
        return deg

    def generate(
        self,
        subgraph: Dict[str, Any],
        image_path: str,
        query_id: str,
    ) -> List[Dict[str, Any]]:
        """Генерирует QA-примеры, отражающие то, что ВИДНО на изображении."""

        real_edges = subgraph["edges"]
        real_nodes = [n["name"] for n in subgraph["nodes"]]
        head = subgraph["head"]
        target_rel = subgraph.get("rel", "?")

        # ── Визуальные данные (то, что на картинке) ───────────
        # +1 узел: [?]
        visual_nodes = real_nodes + [_QUESTION_NODE]

        # +1 ребро: head —[rel]→ [?]
        visual_edges = real_edges + [
            {"source": head, "target": _QUESTION_NODE, "relation": target_rel},
        ]

        # Степени с учётом [?]
        visual_degree = self._compute_degrees(real_edges)
        visual_degree[head] = visual_degree.get(head, 0) + 1  # head→[?]
        visual_degree[_QUESTION_NODE] = 1                      # [?] имеет 1 ребро

        # ── Генерация примеров ────────────────────────────────
        examples: List[Dict[str, Any]] = []

        def _q(task: str) -> str:
            return "<image>\n" + random.choice(self._PROMPTS[task])

        def _add(task: str, answer: str, prompt: str | None = None):
            examples.append({
                "image_path": image_path,
                "prompt": prompt or _q(task),
                "answer": answer,
                "task_type": task,
                "query_id": query_id,
            })

        # 1) Node count (real + [?])
        _add("node_count", str(len(visual_nodes)))

        # 2) Edge count (real + dashed)
        _add("edge_count", str(len(visual_edges)))

        # 3) Node degree — до 3 случайных узлов (включая возможный [?])
        degree_pool = [n for n in visual_nodes if n in visual_degree]
        for node in random.sample(degree_pool, min(3, len(degree_pool))):
            _add(
                "node_degree",
                str(visual_degree[node]),
                prompt=_q("node_degree").format(node=node),
            )

        # 4) Highest degree (с учётом +1 для head)
        if visual_degree:
            max_node = max(visual_degree, key=visual_degree.get)
            _add("highest_degree", max_node)

        # 5) Node listing (включая [?])
        display_nodes = visual_nodes[:10]
        node_list = ", ".join(display_nodes)
        if len(visual_nodes) > 10:
            node_list += "..."
        _add("node_listing", node_list)

        # 6) Triple listing (включая head→[?])
        display_triples = [
            f"({e['source']}, {e['relation']}, {e['target']})"
            for e in visual_edges[:5]
        ]
        triple_str = "; ".join(display_triples)
        if len(visual_edges) > 5:
            triple_str += "; ..."
        _add("triple_listing", triple_str)

        return examples


# ════════════════════════════════════════════════════════════════
# Stage 2 — Link Prediction QA
# ════════════════════════════════════════════════════════════════

class Stage2DataGenerator:
    """Генерирует QA для link prediction. Tail — только в answer."""

    _TEMPLATES: List[str] = [
        (
            "<image>\n"
            "The image shows the neighborhood of entity '{head}' "
            "in a knowledge graph. The dashed edge points to the "
            "unknown entity {question} that you need to predict.\n\n"
            "Entity description: {head_description}\n\n"
            "Query: ({head}, {relation}, ?)\n"
            "Based on the graph structure, predict the missing "
            "tail entity.\n"
            "Answer:"
        ),
        (
            "<image>\n"
            "Knowledge Graph Link Prediction\n\n"
            "The subgraph above shows connections around '{head}'.\n"
            "- Blue node (head) = query entity\n"
            "- Dashed node {question} = entity to predict\n\n"
            "Query: {head} \u2014[{relation}]\u2192 ?\n"
            "Provide only the entity name.\n"
            "Answer:"
        ),
        (
            "<image>\n"
            "Task: Link Prediction\n\n"
            "Context: subgraph centered on '{head}'.\n"
            "The dashed green edge leads to {question} \u2014 "
            "the target entity.\n"
            "Description: {head_description}\n\n"
            "Query: ({head}, {relation}, ?)\n"
            "Predict the tail entity.\n"
            "Answer:"
        ),
    ]

    def generate(
        self,
        head: str,
        relation: str,
        tail: str,
        head_description: str,
        image_path: str,
        query_id: str,
    ) -> Dict[str, Any]:
        """Tail фигурирует ТОЛЬКО в поле answer. Промпт leak-free."""

        prompt = random.choice(self._TEMPLATES).format(
            head=head,
            relation=relation,
            head_description=head_description or head,
            question=_QUESTION_NODE,
        )

        return {
            "image_path": image_path,
            "prompt": prompt,
            "answer": tail,
            "task_type": "link_prediction",
            "query_id": query_id,
            "head": head,
            "relation": relation,
            "tail": tail,
            "head_description": head_description,
        }