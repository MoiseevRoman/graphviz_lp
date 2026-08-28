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
    k_neighbors: int = 12
    use_semantic_weight: bool = True
    min_semantic_weight: float = 0.0
    weight_property: str = "semantic_weight"
    # Добавлять ли рёбра между самими соседями (истинная топология).
    # False → звёздная окрестность (для ablation).
    include_neighbor_edges: bool = True
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
    # head + до 12 соседей + узел [?] → до 14 узлов в subgraph["nodes"].
    max_nodes_display: int = 14
    max_edges_display: int = 25
    max_label_len: int = 30
    node_fontsize: int = 13
    edge_fontsize: int = 11
    head_fontsize: int = 14

    # Включать ли layout-координаты движка в текстовый feature-блок. Дороже —
    # вызывает `dot` на каждый подграф при сборке. degree/centrality считаются
    # всегда (дёшево, чистый networkx). См. text_encoders / graph_features.
    inject_layout_features: bool = False

    # matched-info фикс: ключевать узлы текстовых кодировок ПОЗИЦИОННО (по сырому
    # имени), а не по обрезанной подписи. При коллизии подписей (разные сущности
    # → одинаковый format_label) картинка рисует их разными боксами, а текст
    # матрицы/списка смежности молча схлопывал их. С флагом узлы не схлопываются,
    # а совпавшие подписи дизамбигуируются суффиксом ' #k' (в тексте; при желании
    # то же и в рендере). См. text_encoders._visible_ids. Default False —
    # исторический вывод колонок не меняется.
    disambiguate_labels: bool = False

    @classmethod
    def for_qwen(cls, side: int = 560, **overrides) -> "VisualizationConfig":
        """Пресет рендера под Qwen2.5-VL (dynamic resolution).

        LLaVA жёстко жмёт к 336, поэтому дефолт target_size=336 оптимален.
        Qwen же режет картинку на патчи 28×28, а при площади ниже
        ``qwen_min_pixels`` (256*28*28 ≈ 200k px ≈ сторона 448) АПСКЕЙЛИТ
        низкоразрешённый растр → размытие подписей. Рендерим сразу со стороной
        ≥448 (по умолчанию 560 ≈ 314k px, ближе к середине окна min..max),
        чтобы число визуальных токенов соответствовало реальной детализации,
        а не искусственному увеличению. Шрифты НЕ трогаем: при большем холсте
        и том же кегле на глиф приходится больше пикселей → выше читаемость.
        """
        return cls(target_size=side, **overrides)


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
    # Смена бэкенда = смена ОДНОЙ строки. Поддерживается два семейства:
    #   LLaVA:  "llava-hf/llava-1.5-7b-hf"  (CLIP-336, 576 визуальных токенов)
    #   Qwen2.5-VL-3B: "Qwen/Qwen2.5-VL-3B-Instruct"  (динамич. разрешение)
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    # "auto" → определяется по model_name; можно принудительно "llava" / "qwen".
    model_family: str = "auto"
    model_cache_dir: Optional[str] = None
    output_dir: str = "./graphvis_ckpts"
    resume_from_checkpoint: Optional[str] = None

    # Реализация внимания: None → выбор HF (обычно sdpa/flash).
    # Для attention-проб нужен "eager" (иначе attentions не возвращаются).
    attn_implementation: Optional[str] = None

    # ── Qwen2.5-VL: контроль числа визуальных токенов ──
    # Токенов ≈ (H*W)/(28*28). Для читаемого графа держим ~576 токенов,
    # т.е. окно [min_pixels, max_pixels] вокруг 336×336≈451584 px.
    qwen_min_pixels: int = 256 * 28 * 28   # ~200k px  ≈ 256 токенов
    qwen_max_pixels: int = 720 * 28 * 28   # ~564k px  ≈ 720 токенов

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    # LoRA-таргеты для LLaVA (LM + мультимодальный проектор).
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "multi_modal_projector.linear_1",
            "multi_modal_projector.linear_2",
        ]
    )
    # LoRA-таргеты для Qwen2.5-VL. q/k/v/o_proj есть ТОЛЬКО в LM (ViT
    # использует слитый qkv/proj → не затрагивается). gate/up/down_proj
    # затрагивают и MLP LM, и MLP визуального блока — это допустимо
    # (лёгкие адаптеры на визуальном MLP, ViT-attention остаётся замороженным).
    # Для строго-LM-only варианта оставьте только первые четыре записи.
    lora_target_modules_qwen: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # Разморозить ВИЗУАЛЬНЫЙ путь: навесить LoRA на все Linear зрительной башни
    # (ViT-attention qkv/proj + мультимодальный merger). Без этого замороженный
    # энкодер, обученный на natural images, не кодирует структуру графа, и модель
    # учится игнорировать картинку (real≈blank). Имена модулей обнаруживаются
    # в рантайме (устойчиво к различиям версий Qwen2-VL / 2.5-VL).
    train_vision: bool = False

    # Режим картинки ПРИ ОБУЧЕНИИ: "real" — настоящий рендер (визуальный контекст);
    # "blank"/"noise" — пустой/шумовой холст (для текстовых условий, где окрестность
    # подаётся текстом, а картинка должна быть неинформативной = аналог KGT5-context).
    train_image_mode: str = "real"

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
    # Сторона fallback-картинок (blank/noise/битый файл) в пикселях. ДОЛЖНА
    # совпадать с VisualizationConfig.target_size рендера датасета, иначе
    # blank/real различаются по размеру → разное число визуальных токенов у
    # Qwen → загрязнение абляции модальности (real − blank). Для LLaVA любой
    # размер всё равно жмётся к 336, так что расхождение безвредно, но держим
    # единообразно.
    image_size: int = 336

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