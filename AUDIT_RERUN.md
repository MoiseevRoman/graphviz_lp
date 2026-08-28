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

## Прочие находки аудита (задокументированы как ограничения в статье, раздел 7.2)

- Идентичность по display-имени вместо `fb_id` (омонимы): 223 имени / 499 mid, 6.02% тест-троек.
  Робастный фикс — ключевать сущности по `fb_id` в `retriever`/`eval` (требует полной пересборки).
- Рассогласование очистки имён (loader vs builder): ~439 троек (~2.7%) выпадают из оценки.
- Схлопывание 237→161 отношений (последний сегмент пути): единообразно во всех модальностях.
- Маркер `[?]`: есть в изображении и `neighbors_feat`, нет в `text/adjlist/matrix` — асимметрия в
  пользу визуала (а победил текст), вывод от этого только устойчивее.
