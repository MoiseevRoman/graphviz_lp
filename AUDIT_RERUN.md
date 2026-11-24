# Аудит: применённые фиксы и команды перепрогона

Снимок до правок: git commit `aec386a`. Все изменения смотреть `git diff aec386a`,
откатить всё — `git checkout aec386a -- .`, один файл — `git checkout aec386a -- <file>`.

## Что уже исправлено в коде (за флагами, поведение по умолчанию не меняется)

### Фикс #2 — схлопывание узлов по обрезанной подписи (matched-info)
- `config.py`: новый флаг `VisualizationConfig.disambiguate_labels` (default False).
- `text_encoders.py`: позиционные кодировщики `_visible_ids` + `_encode_{triples,adjlist,matrix}_ids`.
  Узлы ключуются по сырому имени (позиции), совпавшие подписи получают суффикс ` #k`.
- **Перепрогон НЕ требуется.** Проверено на отгруженных данных: `neighbors_matrix`/`neighbors_adjlist`
  собираются `add_text_encodings.py` из ПОЛНЫХ имён `neighbors_text` → коллизий обрезки нет
  (0/4000 подграфов на K4 и K16). Дефект жил только в пути `_visible`/`all_text_prompts`,
  которым отгруженные эксперименты не пользовались. Флаг — защита на будущее.
- Если когда-нибудь пересобираешь датасет через `generators`/`all_text_prompts`, включи флаг:
  `VisualizationConfig.for_qwen(side=560, disambiguate_labels=True)`.

### Фикс #3 — realistic тай-брейк ранга
- `training/evaluate.py` (`evaluate_link_prediction_rank`) и `training/rank_eval.py`:
  параметр `realistic_ties` → ранг `1 + higher + ties/2` (как pykeen и как sample-путь).
  Default False = исторический optimistic (строгое «>»), чтобы старые числа воспроизводились.
- `scripts/viability_gate.py`: флаг `--realistic_ties`.
- `scripts/eval_common.py`: флаг `--realistic_ties` (проброс).
- Влияет только на АБСОЛЮТНЫЕ MRR/Hits (одинаково у всех условий); порядок/вывод не меняется.
  Перепрогон опционален — как проверка робастности абсолютных чисел.

## Команды перепрогона (на Windows-машине с GPU)

Realistic-tie-break на общем тесте — пишем в ОТДЕЛЬНОЕ дерево, старое не трогаем:

    python scripts/eval_common.py ^
      --budgets 1,4,8,16 ^
      --conditions text,text_feat,visual,text_adjlist,text_matrix,both ^
      --eval_n 1000 --eval_mode rank --realistic_ties ^
      --out_root graphvis_ckpts_scaling_common_realistic

    python scripts/summarize_scaling.py ^
      --root graphvis_ckpts_scaling_common_realistic ^
      --out_prefix tmp/common_realistic

(отсутствующие адаптеры условий пропускаются автоматически; сравнить с текущими числами
из `graphvis_ckpts_scaling_common` — порядок условий должен совпасть, абсолютные MRR слегка ниже.)

Aug-условия лежат отдельно — перепрогон при необходимости:

    python scripts/eval_common.py --budgets 4,16 --conditions visual ^
      --ckpt_root graphvis_ckpts_scaling_aug --data_base graphvis_scaling_aug ^
      --eval_n 1000 --eval_mode rank --realistic_ties ^
      --out_root graphvis_ckpts_scaling_aug_realistic

## Исход: k16 text_matrix — отсечение ответа лимитом длины → приводится как н/д (NA)

Прежнее табличное значение text_matrix@K16 = 0.394 бралось из **необученного**
чекпойнта. Перепрогон (`run_fix_text_matrix_k16.bat`, флаги
`--learning_rate/--seed/--max_grad_norm`): LR 2e-4/seed 1 и LR 1e-4/seed 42 — оба
дали абсурдный loss (3e5 / 8e6).

**ИСТИННАЯ ПРИЧИНА (установлена повторным аудитом, первоначальный диагноз
«численная нестабильность bf16+LoRA» — НЕВЕРЕН):**

- `TrainConfig.max_length = 1024` (config.py), а средний промпт text_matrix@K16 =
  **1443 токена**. `families.process_train_example` обрезает по `max_length`, отсекая
  **ответ**; метки маскируются целиком (`labels[:prompt_len] = -100`) → градиента нет.
- Прямое доказательство: в логе перепрогона **3298 предупреждений
  `example truncated: answer fully cut` из 4000 примеров (82.5 %)**.
- Абсурдные значения loss (1e5…1e7) — артефакт нормировки на почти нулевое число
  незамаскированных токенов, а не взрыв градиента.
- Дополнительно: **рассогласование train/eval** — обучение режет на 1024
  (`config.py:215`), оценка на 2048 (`training/evaluate.py:285`).
- Затронут также **text_feat@K16** (952 токена, ≈13 % примеров); его 0.411 идёт в
  таблицы как валидный — это задокументировано в §7.2.
- **text (569 ток.) и visual (468 ток.) не затронуты ни при одном K** → ключевое
  сравнение чистое.

Устранимо без пересборки датасета: поднять `max_length` до 2048 и перепрогнать
только `text_matrix@K16` и `text_feat@K16`. До этого в статье — **н/д** (таблицы 3 и 5, §7.2).

## Нарушение matched-information при K=16 (обрезка рендера)

`visualizer.compute_visible_subgraph` режет `nodes[:max_nodes_display]` (=14) ДО
отрисовки, а `neighbors_text` обрезки не имеет. При K=16 звезда = head + до 16
соседей > 14 узлов:

- **test: 2738/4000 = 68.5 % подграфов, 6799/61236 = 11.1 % рёбер** не попадают на картинку;
- train: 2167/3998 = 54.2 % подграфов, 8.6 % рёбер;
- **K = 1, 4, 8 — ровно 0 % потерь** (head + ≤8 соседей ≤ 14).

Направление — **против визуала**, и только при K=16, где разрыв text−visual и так
статистически незначим (Δ=+0.008, 95 % ДИ [−0.002, +0.018], p=0.126, парный бутстрап
на n=971). Поэтому в статье при K=16 заявляется лишь **отсутствие преимущества
визуала**, а не превосходство текста. Полное устранение требует
`max_nodes_display ≥ 18` + перерендер K16 + переобучение visual (не выполнено).

## Бюджет обучения: iso-example, НЕ iso-token

Все отгруженные прогоны: `train_n = 4000` у всех условий (равное число примеров и
шагов). Токенов на пример при этом различается: text@K1 330, visual 468, text@K16 569,
text_matrix@K16 1443. `iso_budget.json` показывает, что при реальном iso-token было бы
`text=9617, visual=5786` — режим считался, но **не применялся** (флаг `--iso_tokens`
в прогонах не использовался). Формулировка «iso-token» в статье заменена на
**«iso-example»**. Направление: при K ≤ 8 визуал получал БОЛЬШЕ токенов на пример и
всё равно проигрывал → вывод от этого только крепче.

## Прочие находки аудита (задокументированы как ограничения в статье, раздел 7.2)

- Идентичность по display-имени вместо `fb_id` (омонимы): 223 имени / 499 mid, 6.02% тест-троек.
  Робастный фикс — ключевать сущности по `fb_id` в `retriever`/`eval` (требует полной пересборки).
- Рассогласование очистки имён (loader vs builder): ~439 троек (~2.7%) выпадают из оценки.
- Схлопывание 237→161 отношений (последний сегмент пути): единообразно во всех модальностях.
- Маркер `[?]`: есть в изображении и `neighbors_feat`, нет в `text/adjlist/matrix` — асимметрия в
  пользу визуала (а победил текст), вывод от этого только устойчивее.
