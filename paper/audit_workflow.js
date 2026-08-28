export const meta = {
  name: 'audit-graphviz-lp',
  description: 'Three-group audit (inspect -> critique -> fix) of dataset-generation and experiment code',
  phases: [
    { title: 'Проверка' },
    { title: 'Критика' },
    { title: 'Исправление' },
  ],
}

const ROOT = '/workspace'

const CONTEXT = `
Проект graphviz_lp (магистерская, HSE). Исследование = КОНТРОЛИРУЕМОЕ сравнение
модальностей для предсказания связей (link prediction) в графе знаний FB15k-237
с визуально-языковой моделью Qwen2.5-VL. Центральные научные утверждения, которые
код ДОЛЖЕН реально обеспечивать (иначе выводы статьи неверны):
  * matched information: изображение подграфа и его текстовые сериализации
    (edge-list / adjacency-list / adjacency-matrix) кодируют ОДИН И ТОТ ЖЕ набор рёбер;
  * iso-token: обучающий токен-бюджет уравнен между условиями;
  * общий тест: все условия оцениваются на ОДНИХ И ТЕХ ЖЕ тестовых тройках и
    кандидатах, метрики (MRR, Hits@k) сопоставимы;
  * НЕТ утечки ответа: искомый хвост target-тройки не должен попадать в
    визуализируемых/сериализуемых соседей; на валидации видимы только train-рёбра,
    на тесте — train+valid; фильтрованное ранжирование исключает другие истинные хвосты.
Ключевые риски для ревью: утечка данных, некорректный расчёт метрик (тай-брейки,
candidate cap, фильтрация), несовпадение информации между модальностями,
неверный iso-token учёт, недетерминизм оценки, рассогласование частей пайплайна.
Все файлы под ${ROOT}. Читай реальные файлы (абсолютные пути).
`

const MODULES = [
  { label: 'data-load+enrich', files: ['loader.py', 'enricher.py'],
    focus: 'Парсинг FB15k-237, имена/описания сущностей, разбиение на train/valid/test, потенциальные утечки между сплитами, обработка дубликатов/обратных отношений.' },
  { label: 'subgraph+ranking', files: ['builder.py', 'retriever.py'],
    focus: 'Извлечение head-centric подграфа, split-aware фильтрация рёбер (train-only на valid, train+valid на test), ранжирование соседей sentence-BERT top-K, и КРИТИЧНО: не попадает ли истинный target-хвост в отобранные соседи (утечка ответа).' },
  { label: 'generate+render', files: ['generators.py', 'visualizer.py'],
    focus: 'Построение промптов/примеров, рендер Graphviz, соответствие текста и картинки (один и тот же набор рёбер), пометка искомого узла [?] без раскрытия ответа, детерминизм, fallback движков.' },
  { label: 'encodings+features+tokens', files: ['text_encoders.py', 'graph_features.py', 'token_cost.py', 'config.py'],
    focus: 'Текстовые сериализации (edge-list/adjlist/matrix) кодируют те же рёбра; структурные признаки движка; подсчёт токенов (текстовых+визуальных для Qwen по площади) для iso-token и token-cost; дефолты конфигурации.' },
  { label: 'driver:viability_gate', files: ['scripts/viability_gate.py'],
    focus: 'Главный драйвер train+eval: режимы контекста (text/text_feat/text_adjlist/text_matrix/visual/both/...), логика blank-image для текстовых режимов, загрузка адаптера через --adapter_dir, справедливость оценки (--eval_n, общий тест), отсутствие утечки, детерминизм.' },
  { label: 'scaling-orchestration', files: ['scripts/run_scaling.py', 'scripts/scaling_curve.py'],
    focus: 'Оркестрация прогонов по K, расчёт iso-token обучающего бюджета (примеры * токены/пример), согласованность аргументов между условиями, воспроизводимость (seed).' },
  { label: 'common-test+variants', files: ['scripts/eval_common.py', 'scripts/summarize_scaling.py', 'scripts/add_text_encodings.py', 'scripts/relayout_images.py', 'scripts/rerender_dataset.py'],
    focus: 'Общий тест-набор (одинаковые n и тройки для всех условий), matched-info при добавлении текст-кодировок и layout-аугментации (тот же набор рёбер, тест не аугментируется), корректная агрегация report.json в таблицы/CSV/фигуры.' },
  { label: 'metrics:rank-eval', files: ['training/evaluate.py', 'training/rank_eval.py'],
    focus: 'Фильтрованное ранжирование: сбор кандидатов, candidate cap (~50), исключение прочих истинных хвостов (filtered setting), обработка тай-брейков ранга, расчёт MRR/Hits@{1,3,10}, что искомый ответ гарантированно в наборе кандидатов, отсутствие утечки в промпте оценки.' },
  { label: 'train-loop+model', files: ['training/dataset.py', 'training/model.py', 'training/trainer.py', 'training/families.py'],
    focus: 'QLoRA setup, коллация, маскирование лейблов (учится только ответ, не промпт), двухэтапное обучение, целевые модули LoRA, детерминизм/seed.' },
]

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    module: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'короткий слаг' },
          file: { type: 'string', description: 'repo-relative путь, напр. scripts/viability_gate.py' },
          lines: { type: 'string', description: 'диапазон строк, напр. 120-135' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category: { type: 'string', enum: ['leakage', 'metric', 'fairness', 'correctness', 'consistency', 'reproducibility', 'robustness', 'style'] },
          title: { type: 'string' },
          description: { type: 'string', description: 'что не так и почему это важно для выводов' },
          evidence: { type: 'string', description: 'фрагмент кода / рассуждение' },
          affects_reported_results: { type: 'boolean', description: 'изменит ли исправление уже посчитанные числа в статье' },
          suggested_fix: { type: 'string' },
        },
        required: ['file', 'lines', 'severity', 'category', 'title', 'description', 'affects_reported_results', 'suggested_fix'],
      },
    },
  },
  required: ['module', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    real: { type: 'boolean', description: 'это настоящая проблема, а не ложное срабатывание' },
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'needs_refinement'] },
    corrected_severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
    affects_reported_results: { type: 'boolean' },
    safe_to_autofix: { type: 'boolean', description: 'можно ли безопасно применить правку не меняя смысл экспериментов и не ломая уже посчитанные результаты' },
    rationale: { type: 'string' },
    refined_fix: { type: 'string' },
  },
  required: ['real', 'verdict', 'corrected_severity', 'affects_reported_results', 'safe_to_autofix', 'rationale', 'refined_fix'],
}

const FIX_SCHEMA = {
  type: 'object',
  properties: {
    file: { type: 'string' },
    applied: {
      type: 'array',
      items: {
        type: 'object',
        properties: { title: { type: 'string' }, change: { type: 'string' }, lines: { type: 'string' } },
        required: ['title', 'change'],
      },
    },
    skipped: {
      type: 'array',
      items: {
        type: 'object',
        properties: { title: { type: 'string' }, reason: { type: 'string' } },
        required: ['title', 'reason'],
      },
    },
    summary: { type: 'string' },
  },
  required: ['file', 'applied', 'summary'],
}

function inspectPrompt(m) {
  return `${CONTEXT}

Ты — ревьюер ГРУППЫ 1 (проверка правильности и логичности кода). Твой модуль: ${m.label}.
Файлы для глубокого чтения (обязательно прочитай целиком): ${m.files.map(f => ROOT + '/' + f).join(', ')}.
Читай также смежные файлы, если нужно понять контракт (напр. как вызывается функция).

ФОКУС: ${m.focus}

Найди конкретные проблемы КОРРЕКТНОСTИ и ЛОГИКИ (не выдумывай — только то, что реально видно в коде):
приоритет — утечка данных, ошибки метрик, несовпадение информации между модальностями,
нарушение iso-token/общего теста, недетерминизм, рассогласование частей пайплайна.
Для каждой находки укажи точный файл и строки, серьёзность, категорию, чёткое описание
«что не так и почему это ломает выводы», доказательство из кода и предлагаемую правку.
Обязательно честно проставь affects_reported_results (изменит ли фикс уже посчитанные числа).
Если модуль в порядке — верни пустой список findings. Не раздувай список стилистикой.
Верни строго JSON по схеме.`
}

function critiquePrompt(f) {
  return `${CONTEXT}

Ты — критик ГРУППЫ 2 (адекватная критика). Тебе дана ОДНА находка ревьюера. Твоя задача —
скептически перепроверить её по РЕАЛЬНОМУ коду и вынести вердикт. По умолчанию сомневайся:
если это ложное срабатывание, непонимание контракта, недостижимый путь или несущественная
стилистика — ставь real=false / verdict=refuted и объясни почему.

Находка (JSON):
${JSON.stringify({ file: f.file, lines: f.lines, severity: f.severity, category: f.category, title: f.title, description: f.description, evidence: f.evidence, affects_reported_results: f.affects_reported_results, suggested_fix: f.suggested_fix }, null, 2)}

ОБЯЗАТЕЛЬНО открой ${ROOT}/${f.file} на указанных строках (и вокруг) и проверь по факту.
Оцени: (1) реальна ли проблема; (2) правильная ли серьёзность; (3) действительно ли она
влияет на уже посчитанные числа статьи (affects_reported_results) — будь здесь особенно строг;
(4) safe_to_autofix — можно ли применить правку, НЕ меняя смысл экспериментов и НЕ инвалидируя
уже полученные результаты (если фикс меняет поведение, влиявшее на числа, ставь false —
такое нельзя тихо править, только в отчёт). Дай уточнённую правку refined_fix.
Верни строго JSON по схеме.`
}

function fixPrompt(file, items) {
  const list = items.map((x, i) => `#${i + 1} [${x.verdict.corrected_severity}] ${x.title}
  строки: ${x.lines}
  суть: ${x.description}
  правка (уточнённая критиком): ${x.verdict.refined_fix || x.suggested_fix}`).join('\n\n')
  return `${CONTEXT}

Ты — инженер ГРУППЫ 3 (исправление). Файл: ${ROOT}/${file}.
Есть git-снимок проекта, так что твои правки будут просматриваемы как diff — делай МИНИМАЛЬНЫЕ,
точечные изменения строго по подтверждённым находкам. НЕ переформатируй файл, НЕ меняй стиль,
сохраняй язык и плотность комментариев как в окружающем коде (комментарии по-русски, минимум).
НЕ меняй поведение, влияющее на уже посчитанные результаты (такие находки сюда не должны были
попасть; если увидишь такую — пропусти её и укажи в skipped).

Подтверждённые находки для этого файла:
${list}

Действия: прочитай файл, примени правки через Edit, убедись что Python-синтаксис не сломан
(мысленно/логически). Верни строго JSON по схеме: что применил (applied), что пропустил и почему
(skipped), краткое summary.`
}

// ---------- Phase 1: inspect ----------
phase('Проверка')
const inspected = await parallel(MODULES.map(m => () =>
  agent(inspectPrompt(m), { label: `inspect:${m.label}`, phase: 'Проверка', schema: FINDINGS_SCHEMA })
))
const findings = inspected.filter(Boolean).flatMap(r =>
  (r.findings || []).map((f, i) => ({ ...f, module: r.module, id: f.id || `${r.module}-${i}` })))
log(`Группа 1: ${findings.length} находок. Передаю на критику.`)

if (findings.length === 0) {
  return { findings_total: 0, confirmed: 0, safe_fixes: 0, report_only: 0, note: 'находок нет' }
}

// ---------- Phase 2: critique ----------
phase('Критика')
const critiqued = (await parallel(findings.map((f) => () =>
  agent(critiquePrompt(f), { label: `crit:${f.file}#${f.id}`, phase: 'Критика', schema: VERDICT_SCHEMA })
    .then(v => ({ ...f, verdict: v }))
))).filter(Boolean).filter(x => x.verdict)

const confirmed = critiqued.filter(x => x.verdict.real && x.verdict.verdict !== 'refuted')
const refuted = critiqued.filter(x => !(x.verdict.real && x.verdict.verdict !== 'refuted'))
const safe = confirmed.filter(x => x.verdict.safe_to_autofix && !x.verdict.affects_reported_results)
const reportOnly = confirmed.filter(x => !(x.verdict.safe_to_autofix && !x.verdict.affects_reported_results))
log(`Группа 2: подтверждено ${confirmed.length}, опровергнуто ${refuted.length}. К авто-правке ${safe.length}, только в отчёт ${reportOnly.length}.`)

// ---------- Phase 3: fix (one agent per file) ----------
let fixes = []
if (safe.length > 0) {
  phase('Исправление')
  const byFile = {}
  for (const x of safe) { (byFile[x.file] = byFile[x.file] || []).push(x) }
  const files = Object.keys(byFile)
  log(`Группа 3: правлю ${safe.length} находок в ${files.length} файл(ах).`)
  fixes = (await parallel(files.map(file => () =>
    agent(fixPrompt(file, byFile[file]), { label: `fix:${file}`, phase: 'Исправление', schema: FIX_SCHEMA })
  ))).filter(Boolean)
}

const slim = (x) => ({
  file: x.file, lines: x.lines, severity: x.verdict.corrected_severity, category: x.category,
  title: x.title, description: x.description,
  affects_reported_results: x.verdict.affects_reported_results,
  fix: x.verdict.refined_fix || x.suggested_fix, rationale: x.verdict.rationale,
})

return {
  findings_total: findings.length,
  confirmed: confirmed.length,
  refuted: refuted.length,
  safe_fixes: safe.length,
  report_only: reportOnly.length,
  confirmed_details: confirmed.map(slim),
  refuted_details: refuted.map(x => ({ file: x.file, title: x.title, rationale: x.verdict.rationale })),
  report_only_details: reportOnly.map(slim),
  fixes,
}
