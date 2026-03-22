from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional
# from dotenv import load_dotenv

# load_dotenv()

@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.getenv("NEO4J_URI", cls.uri),
            user=os.getenv("NEO4J_USER", cls.user),
            password=os.getenv("NEO4J_PASSWORD", cls.password),
            database=os.getenv("NEO4J_DATABASE", cls.database),
        )


@dataclass
class LoaderConfig:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    data_dir: str = "./FB15k-237"
    entity_names_file: str = "FB15k_mid2name.txt"
    entity_desc_file: str = "FB15k_mid2description.txt"
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    batch_size: int = 2000
    remove_duplicates: bool = True
    
@dataclass
class EnricherConfig:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    node_batch_size: int = 1024
    edge_batch_size: int = 5000
    description_max_len: int = 100

@dataclass
class RetrieverConfig:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    k_neighbors: int = 8
    use_semantic_weight: bool = True
    min_semantic_weight: float = 0.0
    weight_property: str = "semantic_weight"
    subgraph_hop: int = 2
    max_nodes_per_image: int = 25
    max_edges_per_image: int = 40
    remove_duplicate_edges: bool = True


@dataclass
class VisualizationConfig:
    output_dir: str = "./graphvis_fb15k_data"
    engine: str = "dot"
    fmt: str = "png"
    cache_enabled: bool = True

    # ── Целевой размер = вход модели ──
    target_size: int = 336             
    render_dpi: int = 300               

    # ── Адаптивные параметры ──
    max_nodes_display: int = 10
    max_edges_display: int = 15
    max_label_len: int = 15
    node_fontsize: int = 13
    edge_fontsize: int = 11
    head_fontsize: int = 14


@dataclass
class BuilderConfig:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    use_split_property: bool = True
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    sample_size_stage1: Optional[int] = None
    max_triples_for_dataset: Optional[int] = None
    num_visualization_workers: int = 4
    seed: int = 42

    data_dir: str = "./FB15k-237"
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    entity_names_file: str = "FB15k_mid2name.txt"


@dataclass
class TrainConfig:
    # Пути
    dataset_root: str = "./graphvis_fb15k_data/datasets"
    image_root: str = ""
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    model_cache_dir: Optional[str] = None
    output_dir: str = "./graphvis_ckpts"
    resume_from_checkpoint: Optional[str] = None

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "multi_modal_projector.linear_1",
            "multi_modal_projector.linear_2",
        ]
    )

    # Learning rate — единый для всех параметров
    learning_rate: float = 2e-4

    # Optimizer / Scheduler
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Batch
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2

    # Length / Image
    max_length: int = 1024
    pad_image_to_square: bool = True
    group_by_length: bool = False

    # Schedule
    stage1_num_epochs: int = 1
    stage2_num_epochs: int = 2
    stage1_max_samples: Optional[int] = None
    stage2_max_samples: Optional[int] = None

    # Precision
    use_4bit: bool = True
    use_bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging / Saving
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 1000
    save_total_limit: int = 3

    # Dataloader
    dataloader_num_workers: int = 16
    dataloader_pin_memory: bool = True

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.005

    # Колонки датасета
    image_column: str = "image_path"
    prompt_column: str = "prompt"
    answer_column: str = "answer"

    # Evaluation
    eval_num_beams: int = 10
    eval_max_new_tokens: int = 32
    eval_max_samples: int = 1000
    eval_batch_size: int = 16
    eval_image_workers: int = 16