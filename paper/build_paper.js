const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, ImageRun, PageBreak
} = require("docx");

const CW = 9026;            // content width (A4, 1" margins)
const FONT = "Times New Roman";

// ---------- helpers ----------
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });

function P(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun(text)];
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: 120, line: 276 },
    indent: opts.noIndent ? undefined : { firstLine: 480 },
    children: runs,
  });
}
const B = (t) => new TextRun({ text: t, bold: true });
const R = (t) => new TextRun(t);
const I = (t) => new TextRun({ text: t, italics: true });

function bullet(t) {
  return new Paragraph({
    numbering: { reference: "bul", level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60, line: 276 },
    children: Array.isArray(t) ? t : [new TextRun(t)],
  });
}
function num(t, ref = "num") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60, line: 276 },
    children: Array.isArray(t) ? t : [new TextRun(t)],
  });
}

const bd = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
function cell(text, w, { head = false, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align,
      spacing: { after: 0, line: 240 },
      children: [new TextRun({ text: String(text), bold: head, size: 19 })],
    })],
  });
}
function table(widths, rows) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, ri) =>
      new TableRow({ children: r.map((c) => cell(c, widths[r.indexOf(c) === -1 ? 0 : r.indexOf(c)], { head: ri === 0, align: typeof c === "number" || ri === 0 ? AlignmentType.CENTER : AlignmentType.LEFT })) })),
  });
}
// robust table builder (avoid indexOf pitfalls)
function tbl(widths, rows, { firstColLeft = true } = {}) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, ri) => new TableRow({
      children: row.map((val, ci) => cell(val, widths[ci], {
        head: ri === 0,
        align: ri === 0 ? AlignmentType.CENTER : (ci === 0 && firstColLeft ? AlignmentType.LEFT : AlignmentType.CENTER),
      })),
    })),
  });
}
function caption(t) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 160 },
    children: [new TextRun({ text: t, italics: true, size: 19 })],
  });
}
function figure(path, wPx, hPx, cap) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(path),
        transformation: { width: wPx, height: hPx },
        altText: { title: cap, description: cap, name: cap },
      })],
    }),
    caption(cap),
  ];
}
function ref(n, text) {
  return new Paragraph({
    spacing: { after: 40, line: 240 },
    indent: { left: 480, hanging: 480 },
    children: [new TextRun({ text: `[${n}] `, bold: true, size: 19 }), new TextRun({ text, size: 19 })],
  });
}

// ---------- document ----------
const children = [];

// Title block
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "НИУ «Высшая школа экономики» · Факультет компьютерных наук", size: 20 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: "Образовательная программа «Искусственный интеллект»", size: 20 })] }));

children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ bold: true, size: 30,
    text: "Действительно ли визуализация окрестности графа помогает визуально-языковым моделям в предсказании связей? Контролируемое сравнение модальностей при выровненной информации и вычислительном бюджете" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ italics: true, size: 22,
    text: "Does Neighborhood Visualization Help Vision-Language Models in Link Prediction? A Controlled Modality Comparison under Matched Information and Compute" })] }));

children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 },
  children: [new TextRun({ text: "Моисеев Роман Романович", size: 22 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Научный руководитель: к.ф.-м.н., доцент Е. О. Канотистова · Соруководитель: эксперт М. А. Хажгериев", size: 18 })] }));

// Abstract RU
children.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Аннотация", bold: true, size: 24 })] }));
children.push(P([
  R("Неполнота графов знаний (до 60–93 % отсутствующих истинных связей в крупнейших графах) ограничивает их применение в системах генерации с извлечением (RAG). Ранее нами и в ряде недавних работ предлагалось подавать окрестность головной вершины в визуально-языковую модель (LVLM) в виде "),
  I("изображения графа"),
  R("; отдельные публикации (GITA, ViSR-KGC) сообщают о превосходстве визуальной модальности над текстом. Однако эти сравнения не контролируют ни объём подаваемой информации, ни вычислительный бюджет и сопоставляют визуал со "),
  I("слабым"),
  R(" текстовым бейзлайном. В данной работе мы проводим первое "),
  B("контролируемое"),
  R(" сравнение модальностей для задачи предсказания связей на бенчмарке FB15k-237 с моделью Qwen2.5-VL. При одинаковом наборе рёбер во всех модальностях (matched information), равном обучающем токен-бюджете (iso-token) и едином тестовом наборе мы сравниваем: изображение подграфа, текстовые сериализации (список рёбер, список смежности, матрица смежности), текст с явными структурными признаками рендер-движка и их комбинацию. Мы строим кривую масштабирования (качество против числа объектов K) с наложенной токен-стоимостью и даём визуальной модальности её сильнейшую версию через layout-аугментацию — ключевой приём GITA. "),
  B("Результат: "),
  R("обычный текст (список рёбер) даёт лучшее качество при наименьшей стоимости; визуальная модальность стабильно слабее и не добавляет ничего поверх текста даже с аугментацией раскладок; предвычисленные движком признаки избыточны. Мы показываем, почему предшествующие работы систематически переоценивали визуальную модальность."),
]));
children.push(P([B("Ключевые слова: "), R("предсказание связей, граф знаний, визуально-языковая модель, сравнение модальностей, визуальное представление подграфа, эффективность по токенам.")], { noIndent: true }));
children.push(P([B("Код: "), R("https://github.com/MoiseevRoman/graphviz_lp")], { noIndent: true }));

// Abstract EN
children.push(new Paragraph({ spacing: { before: 120, after: 80 }, children: [new TextRun({ text: "Abstract", bold: true, size: 24 })] }));
children.push(P([
  R("Knowledge-graph incompleteness limits retrieval-augmented generation. It has been proposed to feed the neighborhood of a head entity to a vision-language model (LVLM) as a rendered graph image, and recent work (GITA, ViSR-KGC) reports that the visual modality outperforms text. These comparisons, however, control neither the amount of information nor the compute budget, and pit vision against a weak text baseline. We present the first "),
  I("controlled"),
  R(" modality comparison for link prediction on FB15k-237 with Qwen2.5-VL. Holding the edge set identical across modalities (matched information), equalizing the training token budget (iso-token), and evaluating on a shared test set, we compare a subgraph image, textual serializations (edge list, adjacency list, adjacency matrix), text augmented with explicit rendering-engine structural features, and their combination. We plot a scaling curve (quality vs. number of objects K) with token cost overlaid, and give the visual modality its strongest form via layout augmentation — GITA’s key lever. "),
  B("Findings: "),
  R("plain text (edge list) yields the best quality at the lowest cost; the visual modality is consistently weaker and adds nothing on top of text even with layout augmentation; engine-precomputed features are redundant. We explain why prior work overestimated the visual modality."),
]));

children.push(new Paragraph({ children: [new PageBreak()] }));

// I. Introduction
children.push(H1("1. Введение"));
children.push(P([
  R("Граф знаний (KG) — структура данных из троек «(головная сущность, отношение, хвостовая сущность)» "),
  R("(h, r, t), используемая в вопросно-ответных и RAG-системах [1, 2]. Даже в крупнейших графах — Freebase [3], Wikidata [4] — по оценкам [5, 6] отсутствует от 60 % до 93 % истинных связей. Неполнота напрямую вредит RAG: неполный внешний источник ведёт к отсутствию обоснования фактов и галлюцинациям языковых моделей [7]. Задача предсказания связей — восстановление недостающей хвостовой сущности в тройке (h, r, ?)."),
]));
children.push(P([
  R("Существующие подходы делятся на четыре парадигмы. "),
  B("Эмбеддинговые"), R(" методы (TransE [8], DistMult [9], ComplEx [10], RotatE [11]) обучают векторы сущностей и отношений, но игнорируют текстовую семантику и плохо работают на новых сущностях. "),
  B("Графовые"), R(" методы (R-GCN [12], CompGCN [13]) агрегируют окрестность, но плохо обобщаются на редкие отношения. "),
  B("Текстовые"), R(" методы (KG-BERT [14], SimKGC [15], KGT5 [16], KGT5-context [17], KG-LLM [18]) кодируют граф линеаризацией троек в текст, теряя пространственную структуру окрестности. "),
  B("Визуально-языковой"), R(" подход — предмет данной работы — подаёт окрестность как изображение графа."),
]));
children.push(P([
  R("Мотивация визуальной модальности опирается на концепцию «визуального интерфейса как универсального представления» (Карпатый [21]) и на результаты PIXEL [19] (языковое моделирование по пикселям) и Pix2Struct [20] (извлечение структуры из скриншота). Недавние работы применили её к графам: GITA [34] сообщает о выигрыше визуала на задачах графового ризонинга; GVN/E-GVN [35] встраивают визуальные структурные признаки в графовые сети; ViSR-KGC [36] строит мультимодальный конвейер для пополнения графа. Все они формируют впечатление, что "),
  I("визуальная модальность несёт структуру, недоступную тексту."),
]));
children.push(P([
  B("Проблема. "),
  R("Эти сравнения "),
  B("не контролируют"),
  R(" ключевые факторы: (i) объём информации в модальностях (изображение и текст описывают разные подграфы, либо изображение дополнено фотографиями сущностей); (ii) вычислительный бюджет (визуал дообучается, текст — нередко zero-shot; визуал усиливается layout-аугментацией, текст — нет); (iii) силу текстового бейзлайна (обычно наивная линеаризация, в которой LLM заведомо слабы). Поэтому неясно, есть ли у визуальной модальности "),
  I("собственное"),
  R(" преимущество, или наблюдаемый выигрыш — артефакт неравного сравнения."),
]));
children.push(P([
  B("Вопрос исследования. "),
  R("Несёт ли визуальная модальность информацию сверх текста для предсказания связей, "),
  B("когда информация и вычислительный бюджет выровнены"),
  R(", а текстовый бейзлайн взят в сильнейшей форме?"),
]));
children.push(P([
  B("Гипотеза. "),
  R("Изображение подграфа — это повторное кодирование тех же рёбер, что доступны тексту; предвычисленные движком структурные признаки (центральность через позицию/размер) можно выписать и текстом. Следовательно, при выровненных условиях визуальная модальность "),
  I("не даст"),
  R(" преимущества над сильным текстовым представлением."),
]));
children.push(P([B("Вклад работы."), R("")], { noIndent: true }));
children.push(num([B("Контролируемый протокол"), R(" сравнения модальностей: одинаковый набор рёбер (matched information), равный обучающий токен-бюджет (iso-token) и единый тестовый набор.")]));
children.push(num([B("Кривая масштабирования"), R(" качества против числа объектов K с наложенной токен-стоимостью инференса.")]));
children.push(num([B("Сильнейшие бейзлайны обеих сторон"), R(": несколько текстовых сериализаций (список рёбер / смежности / матрица) и сильнейший визуал через layout-аугментацию (приём GITA).")]));
children.push(num([B("Контролируемый вывод"), R(": текст ≥ визуал на всех масштабах, движковые признаки избыточны, изображение не добавляет ничего поверх текста; и объяснение, почему литература переоценивала визуал.")]));
children.push(P([R("Далее: раздел 2 — обзор литературы и позиционирование; раздел 3 — метод и контролируемый протокол; раздел 4 — установка экспериментов; раздел 5 — результаты; раздел 6 — обсуждение; раздел 7 — ограничения; раздел 8 — заключение.")]));

// II. Related work
children.push(H1("2. Обзор литературы"));
children.push(H2("2.1. Формализация задачи"));
children.push(P("Граф знаний G = (V, E), где V — сущности, E — типизированные рёбра. Предсказание связей — ранжирование сущностей-кандидатов по правдоподобию тройки (h, r, t) для запроса (h, r, ?). Качество измеряется MRR (средний обратный ранг правильного ответа) и Hits@k (доля запросов с правильным ответом в топ-k)."));
children.push(H2("2.2. Эмбеддинговые и графовые методы"));
children.push(P([
  R("Эмбеддинговые методы обучают скоринг f(h, r, t): TransE [8] моделирует отношение как сдвиг, но не выражает связи «один-ко-многим»; DistMult [9] симметричен; ComplEx [10] вводит комплексное пространство для асимметрии; RotatE [11] моделирует отношение как поворот. Показано [25], что при аккуратном подборе гиперпараметров эти простые модели уже конкурентоспособны — дальнейший прогресс требует "),
  I("новых источников информации"),
  R(" (текстовой семантики или, как здесь, структурного контекста). Их ограничение — работа с идентификаторами и «холодный старт» на новых сущностях. Графовые сети R-GCN [12] и CompGCN [13] учитывают окрестность, но ненадёжны на редких отношениях."),
]));
children.push(H2("2.3. Текстовые методы"));
children.push(P("KG-BERT [14] применил BERT к тройкам, но требовал прохода на каждого кандидата; SimKGC [15] перешёл к bi-encoder с контрастивным обучением; KGT5 [16] переформулировал задачу как seq2seq-генерацию имени хвоста; KGT5-context [17] добавил текстовое описание 1-hop окрестности в промпт, существенно подняв качество — это ближайший текстовый аналог нашего контекста. Talk-like-a-Graph [37] показал, что выбор способа сериализации графа в текст (список рёбер, смежность и др.) меняет качество LLM на графах на десятки процентов — важный факт для честного текстового бейзлайна. Общее ограничение: работа в текстовом пространстве без явной пространственной структуры."));
children.push(H2("2.4. Визуально-языковые модели и визуальные подходы к графам"));
children.push(P("LLaVA [28] и LLaVA-1.5 [23] соединяют визуальный энкодер CLIP ViT-L/14 [29] с языковой моделью через проектор; Qwen2.5-VL поддерживает нативное динамическое разрешение, что делает число визуальных токенов зависящим от площади изображения. LoRA [31] и QLoRA [24] обеспечивают параметрически эффективное дообучение (обучается <1 % весов)."));
children.push(P([
  R("Визуальные подходы к графам: "),
  B("GITA [34]"), R(" рендерит граф в изображение и решает задачи графового ризонинга (циклы, связность, кратчайший путь); ключевой их вывод — layout-аугментация даёт наибольший прирост качества визуала. "),
  B("GVN/E-GVN [35]"), R(" извлекают «визуальные структурные признаки» замороженным энкодером и вливают их в графовую сеть; их метрика reproduction ratio показывает, что эти признаки во многом "),
  I("воспроизводят рукотворные"),
  R(" структурные фичи. "),
  B("ViSR-KGC [36]"), R(" строит query-aware подграф, рендерит его (лучшая раскладка — dot) и вместе с фотографиями сущностей и списком кандидатов подаёт в VLM; их собственная абляция отмечает, что изображение подграфа «менее критично, чем текст»."),
]));
children.push(H2("2.5. Позиционирование данной работы"));
children.push(P("Все перечисленные визуальные работы либо решают задачи иного типа (структурные головоломки на абстрактных графах — GITA), либо используют визуал как дополнительный признак поверх сильной графовой сети (GVN), либо нагружают визуальное условие дополнительной информацией — реальными фотографиями сущностей и кандидатами из представленческого обучения (ViSR-KGC). Ни одна из них не уравнивает информацию и вычислительный бюджет и не сравнивает визуал с сильнейшим текстовым бейзлайном. Мы закрываем именно этот пробел."));
children.push(caption("Таблица 1. Позиционирование парадигм."));
children.push(tbl([2400, 1500, 1500, 1700, 1926], [
  ["Свойство", "Эмбеддинги", "Графовые", "Текст (KGT5-ctx)", "Наш контроль"],
  ["Новые сущности", "нет", "частично", "да", "да"],
  ["Структура окрестности", "неявно", "да", "линеаризация", "явно (все модальности)"],
  ["Matched information", "—", "—", "—", "да"],
  ["Iso-token обучение", "—", "—", "—", "да"],
  ["Общий тест-набор", "—", "—", "—", "да"],
]));

// III. Method
children.push(H1("3. Метод"));
children.push(H2("3.1. Конвейер построения контекста"));
children.push(P("Тройки всех сплитов загружаются в графовую СУБД Neo4j; каждое ребро хранит метку сплита, что исключает утечку (при валидации доступен только train, при тесте — train+valid). Для каждого запроса (h, r, ?) извлекается head-centric подграф. Из-за ограничения по пикселям число визуализируемых соседей ограничено: соседи ранжируются по косинусной близости текстовых эмбеддингов sentence-BERT [22] (косинус даёт больший эффективный диапазон, чем TransE-скоринг на тех же нормированных эмбеддингах), и берутся top-K."));
children.push(...figure("paper/fig_example.png", 250, 250,
  "Рисунок 1. Пример визуального контекста: head-centric подграф (голова — синий прямоугольник, соседи — жёлтые, пунктирное зелёное ребро ведёт к искомому хвосту [?]). Тот же набор рёбер подаётся текстом в текстовых условиях (matched information)."));
children.push(P([
  R("Изображение рендерится Graphviz (движок dot). "),
  B("Ключевой принцип контроля: "),
  R("во всех модальностях используется "),
  I("один и тот же набор рёбер"),
  R(" и единые поверхностные формы имён (унифицированные подписи), так что текст и изображение описывают идентичный подграф. Дообучение — QLoRA [24, 31] (4-битные замороженные веса, обучаемые адаптеры на векторах внимания, MLP и мультимодальном проекторе). В исходной постановке предусмотрено двухэтапное обучение (сначала понимание графовой структуры, затем целевая задача); контролируемые эксперименты ниже проводятся в одностадийном режиме для чистоты бюджета."),
]));
children.push(H2("3.2. Условия сравнения модальностей"));
children.push(P("Все условия делят один подграф и различаются только представлением контекста:"));
children.push(bullet([B("visual"), R(" — изображение подграфа + минимальный текстовый запрос.")]));
children.push(bullet([B("text (edge-list)"), R(" — список рёбер «(источник, отношение, цель)».")]));
children.push(bullet([B("text_adjlist / text_matrix"), R(" — те же рёбра как список смежности / матрица смежности с легендой отношений.")]));
children.push(bullet([B("text_feat"), R(" — текст рёбер + явные структурные признаки движка (степень, PageRank, betweenness, координаты раскладки).")]));
children.push(bullet([B("both"), R(" — изображение + текст рёбер одновременно (проверка: добавляет ли картинка что-то поверх текста).")]));
children.push(H2("3.3. Контроль информации и вычислений"));
children.push(P([
  B("Matched information: "), R("одинаковые рёбра и подписи во всех модальностях. "),
  B("Iso-token: "), R("обучающий токен-бюджет уравнивается между условиями (число обучающих примеров подбирается так, чтобы произведение «примеры × токены/пример» совпадало), поэтому «равные ресурсы» относятся именно к обучению. "),
  B("Общий тест: "), R("все условия оцениваются на одной и той же seed-фиксированной выборке тестовых троек (одинаковые тройки и кандидаты для всех условий и всех K), что делает MRR сопоставимыми. "),
  B("Токен-стоимость: "), R("для каждого условия измеряется среднее число входных токенов (текстовые + визуальные; визуальные для Qwen считаются по площади изображения)."),
]));
children.push(P([
  B("Сильнейший визуал (layout-аугментация). "),
  R("Чтобы сравнение было честным и к визуалу, мы даём ему его лучшую по GITA [34] версию: обучающие изображения рендерятся разными раскладками (dot/neato/fdp/circo/twopi, по движку на пример), тест — фиксированным dot. Набор рёбер при этом не меняется — меняется только геометрия раскладки."),
]));

// IV. Experimental setup
children.push(H1("4. Установка экспериментов"));
children.push(P([
  R("Данные — FB15k-237 [33] (310 079 троек, 14 505 сущностей, 237 отношений), с именами и описаниями из Freebase [3]. Модель — "),
  B("Qwen2.5-VL-3B-Instruct"),
  R(", дообучение QLoRA, 1 эпоха. Оценка — фильтрованное ранжирование кандидатов (rank; до 50 кандидатов на запрос), метрики MRR и Hits@{1,3,10}. Число объектов (соседей) варьируется K ∈ {1, 4, 8, 16}. Все условия оценены на общем тест-наборе; фактически оценено n = 971 тройка на условие (одинаковые тройки во всех ячейках). Токен-стоимость измеряется на выборке запросов."),
]));

// V. Results
children.push(H1("5. Результаты"));
children.push(H2("5.1. Текст против изображения по масштабам"));
children.push(caption("Таблица 2. Сравнение модальностей на общем тесте (MRR / Hits@1 / Hits@10; ток — среднее число входных токенов на запрос)."));
children.push(tbl([900, 2050, 2050, 2050, 1976], [
  ["K", "text (edge-list)", "text_feat", "visual", "Δ(text−visual), MRR"],
  ["1", "0.398 / 29.4 / 61.4 · 330", "0.396 / 28.5 / 61.3 · 411", "0.380 / 26.9 / 59.6 · 468", "+0.018"],
  ["4", "0.406 / 29.8 / 62.9 · 380", "0.406 / 30.2 / 63.1 · 536", "0.381 / 27.1 / 59.7 · 468", "+0.026"],
  ["8", "0.406 / 29.2 / 63.1 · 445", "0.407 / 29.5 / 63.0 · 695", "0.388 / 27.9 / 62.0 · 468", "+0.017"],
  ["16", "0.412 / 29.9 / 65.3 · 569", "0.411 / 30.0 / 64.3 · 952", "0.404 / 29.2 / 63.6 · 468", "+0.008"],
]));
children.push(P([
  R("Текст превосходит изображение на всех K. Текст "),
  I("масштабируется"),
  R(" с числом объектов (MRR 0.398→0.412), тогда как токен-стоимость визуала "),
  I("постоянна"),
  R(" (468): изображение фиксированного размера при росте K лишь уплотняет содержимое, а не добавляет информацию. Визуал растёт с K (0.380→0.404) и сокращает разрыв, но нигде не обгоняет текст. Признаки движка (text_feat) неотличимы от текста при вдвое-втрое большей стоимости."),
]));
children.push(...figure("paper/fig_scaling.png", 560, 205,
  "Рисунок 2. Кривая масштабирования: MRR, Hits@10 и входные токены против числа объектов K для text / text_feat / visual (общий тест)."));

children.push(H2("5.2. Несколько текстовых представлений (сильнейший текстовый бейзлайн)"));
children.push(caption("Таблица 3. Текстовые сериализации одного подграфа (MRR · токены)."));
children.push(tbl([900, 2031, 2031, 2032, 2032], [
  ["K", "edge-list", "adjacency-list", "adjacency-matrix", "text_feat"],
  ["4", "0.406 · 380", "0.403 · 395", "0.406 · 523", "0.406 · 536"],
  ["16", "0.412 · 569", "0.412 · 593", "0.394 · 1443", "0.411 · 952"],
]));
children.push(P("Список рёбер (edge-list) — уже оптимальная текстовая сериализация: список смежности эквивалентен, а матрица смежности проваливается на больших K (0.394) при кратно большей стоимости (1443 токена). Значит, преимущество текста — не артефакт слабой сериализации: даже сильнейший текст остаётся дешёвым списком рёбер."));

children.push(H2("5.3. Добавляет ли изображение что-то поверх текста? Роль layout-аугментации"));
children.push(caption("Таблица 4. Комбинация модальностей и сильнейший визуал (MRR). *_aug — обучение с layout-аугментацией."));
children.push(tbl([1200, 1956, 1956, 1957, 1957], [
  ["K", "text", "both", "visual", "visual_aug / both_aug"],
  ["4", "0.406", "0.404", "0.381", "0.384 / 0.409"],
  ["16", "0.412", "0.410", "0.404", "0.381 / 0.409"],
]));
children.push(P([
  B("Изображение не добавляет ничего поверх текста: "),
  R("both ≈ text на обоих масштабах (K=16: both 0.410 против text 0.412) при большей стоимости. "),
  B("Layout-аугментация не помогает: "),
  R("для visual прирост отсутствует (K=4: +0.003; K=16: −0.023), для both — тоже. То есть даже дав визуалу его лучшую по GITA версию, мы не получаем преимущества над текстом."),
]));
children.push(...figure("paper/fig_verdict.png", 560, 214,
  "Рисунок 3. Слева — кривая MRR(K) для text/text_feat/visual. Справа — все условия при K=16 (подписи: MRR и токены)."));

// VI. Discussion
children.push(H1("6. Обсуждение: почему в предыдущих работах визуал казался лучше"));
children.push(P([B("Тип задачи. "), R("Выигрыш GITA получен на структурно-алгоритмических задачах (циклы, связность) на абстрактных пронумерованных графах, где ответ — это видимая геометрия, а текстовому LLM приходится последовательно обходить список рёбер. Предсказание связей на FB15k-237 семантическое (нужно знание о сущностях), и «увидеть структуру» здесь не помогает.")]));
children.push(P([B("Что содержит «картинка». "), R("В ViSR-KGC изображение включает реальные фотографии сущностей (мультимодальный граф) — подлинно дополнительную информацию, которой нет в тексте; в GITA картинка — единственный носитель топологии. В нашей постановке изображение — это повторное кодирование тех же рёбер, что и в тексте, поэтому уникальной информации в нём нет, и прироста нет.")]));
children.push(P([B("Визуал как дополнительный канал, а не как модальность. "), R("В GVN визуальные признаки приклеены к сильной графовой сети и, по их же метрике, во многом воспроизводят рукотворные структурные фичи; это «ещё один признак помогает», а не «визуал > текст».")]));
children.push(P([B("Неконтролируемое сравнение и слабый текст. "), R("Во всех трёх работах текстовый бейзлайн — наивная линеаризация (в которой LLM слабы), а визуальное условие дообучено и/или усилено (аугментация, фотографии, кандидаты). Когда обеим сторонам дают лучший шанс и уравнивают вход и обучающий бюджет, мнимое преимущество визуала исчезает. Примечательно, что даже тонкая абляция ViSR-KGC («изображение менее критично, чем текст») согласуется с нашим выводом.")]));

// VII. Limitations & threats to validity
children.push(H1("7. Ограничения и угрозы валидности"));
children.push(H2("7.1. Ограничения охвата"));
children.push(bullet("Одна семья моделей (Qwen2.5-VL-3B), один бенчмарк (FB15k-237), rank-оценка, 1 эпоха обучения."));
children.push(bullet("Изображение — диаграмма подграфа без фотографий сущностей; вывод не переносится на мультимодальные графы с реальными изображениями сущностей."));
children.push(bullet([R("Визуал сокращает разрыв с ростом K — на очень плотных/больших графах, где текст становится чрезмерно длинным, возможен "), I("кроссовер"), R("; проверка — направление будущей работы.")]));
children.push(bullet("Вывод локализует источник выигрыша прежних работ, но не утверждает бесполезность визуальной модальности во всех задачах."));

children.push(H2("7.2. Угрозы валидности реализации"));
children.push(P([
  R("Аудит конвейера выявил ряд артефактов реализации, которые мы фиксируем прозрачно. Ключевой момент: ни один из них не переворачивает вывод, потому что каждый действует "),
  B("единообразно на все условия"),
  R(" (и потому сокращается при вычитании), либо "),
  B("асимметрично в пользу визуальной стороны"),
  R(" (а победил всё равно текст). Масштабы измерены на исходных данных FB15k-237 и на самих собранных датасетах."),
]));
children.push(caption("Таблица 5. Угрозы валидности, их измеренный масштаб и направление влияния на сравнение модальностей."));
children.push(tbl([3050, 1650, 4326], [
  ["Угроза", "Масштаб", "Направление влияния на вывод"],
  ["Идентичность сущностей по display-имени, а не по стабильному mid (омонимы): подграф, фильтрованное ранжирование и исключение хвоста работают по схлопнутой идентичности", "223 имени / 499 mid; 1233/20466 = 6.02% тест-троек", "единообразно на все условия; прямой утечки ответа нет (хвост исключается по имени) — измерительный шум, не смещение в пользу модальности"],
  ["Рассогласование очистки имён (хранение без апострофов/кавычек vs поиск по сырому имени) → часть голов не находится и выпадает из оценки", "≈439 троек (≈2.7%)", "единообразно на все условия; уменьшает эффективный размер теста, не даёт преимущества ни одной модальности"],
  ["Схлопывание 237 отношений в 161 (берётся последний сегмент пути); показывается одинаково во всех модальностях и согласуется с запросом", "237→161; 38 тест-троек совпадают с valid", "единообразно; снижает абсолютную информативность задачи, не раскрывает ответ и не выделяет модальность"],
  ["Маркер цели: изображение и text_feat несут узел [?] и пунктирное ребро head→[?]; edge-list/adjlist/matrix передают запрос только строкой Query", "[?] в neighbors_feat 4000/4000, в text/adjlist/matrix 0/4000", "асимметрия В ПОЛЬЗУ визуала/feat; тем не менее голый текст не уступает — вывод усиливается, а не ослабляется"],
  ["Латентная коллизия подписей в кодировщике text_encoders._visible (обрезка format_label). Проверка отгруженных колонок: neighbors_matrix/adjlist строятся из полных имён (add_text_encodings) — дефект в этот путь не попал", "0/4000 подграфов с коллизией на K4 и K16 (измерено)", "нулевое влияние на числа: text_matrix/text_adjlist оценивались на бесколлизийных колонках; провал матрицы на K=16 (0.394) — настоящий результат. Дефект устранён защитно за флагом"],
  ["Оптимистичный тай-брейк ранга (строгое «>»): при равенстве очков золотой ставится на лучшую позицию", "все запросы rank-режима", "единообразно завышает абсолютные MRR/Hits у ВСЕХ условий; относительное сравнение (утверждение статьи) не затронуто"],
  ["Детерминизм оценки: headline — фильтрованный rank (детерминирован); sample/KGT5-числа, где приводятся, — на одном сиде", "sample-режим", "не влияет на headline-таблицу (rank-режим)"],
]));
children.push(P([
  R("Направление важнейших дефектов либо нейтрально (омонимы, тай-брейк, отсев по пунктуации, схлопывание отношений — действуют на все условия сразу), либо благоприятно для визуальной модальности (маркер [?] есть у картинки, но не у edge-list, который и выиграл). Поэтому контролируемый вывод "),
  I("текст ≥ визуал"),
  R(" устойчив к этим артефактам. Прямая проверка отгруженных данных подтвердила отсутствие коллизий подписей в текстовых кодировках (0 из 4000 подграфов), поэтому все приведённые числа text_matrix/text_adjlist остаются в силе, а провал матрицы на больших K — содержательный, а не артефактный. Единственный дефект, способный сдвинуть "),
  I("абсолютные"),
  R(" (но не относительные) значения — оптимистичный тай-брейк ранга: перепрогон с realistic-правилом (реализован за флагом) понизил бы MRR/Hits у всех условий на одинаковую долю ничьих, не меняя порядок. Остальные пункты либо устранены защитно в коде, либо задокументированы как ограничения охвата.")
]));
children.push(H3("Проверка чувствительности к идентичности по имени"));
children.push(P([
  R("Чтобы убедиться, что важнейшая угроза (омонимы) не определяет результат, мы пересчитали метрики на "),
  B("неомонимном подмножестве"),
  R(" — исключив все запросы, где голова или хвост имеют неоднозначное имя (n = 920 из 971). Вывод "),
  I("текст ≥ визуал"),
  R(" сохраняется на всех K, а разрыв даже слегка "),
  B("увеличивается"),
  R(" (ΔMRR text−visual: K=1 +0.018→+0.021; K=4 +0.026→+0.028; K=8 +0.017→+0.020; K=16 +0.008→+0.010). Абсолютные числа text практически не меняются (напр. K=16: 0.412→0.412), visual чуть ниже (0.404→0.402); провал text_matrix на K=16 сохраняется (0.394→0.392). Следовательно, идентификация сущностей по имени не является драйвером наблюдаемого преимущества текста.")
]));

// VIII. Conclusion
children.push(H1("8. Заключение"));
children.push(P("Мы провели первое контролируемое сравнение модальностей для предсказания связей: при выровненной информации (одинаковые рёбра), равном обучающем токен-бюджете и общем тесте обычный текст (список рёбер) даёт лучшее качество при наименьшей стоимости; визуальная модальность стабильно слабее и не добавляет ничего поверх текста даже с layout-аугментацией; предвычисленные движком структурные признаки избыточны. Наблюдаемое в литературе превосходство визуала объясняется типом задачи, реально дополнительной информацией (фотографии сущностей) или ролью визуала как признакового канала, а также неконтролируемым сравнением со слабым текстом. Это контролируемый результат, объясняющий, почему область систематически переоценивала визуальную модальность, и задающий корректный протокол для будущих сравнений."));

// References
children.push(H1("Список использованных источников"));
const refs = [
  "S. Ji et al., “A Survey on Knowledge Graphs,” IEEE TNNLS, 33(2):494–514, 2022.",
  "Q. Wang et al., “Knowledge Graph Embedding: A Survey,” IEEE TKDE, 29(12):2724–2743, 2017.",
  "K. Bollacker et al., “Freebase,” ACM SIGMOD, 2008, pp. 1247–1250.",
  "D. Vrandečić, M. Krötzsch, “Wikidata,” Commun. ACM, 57(10):78–85, 2014.",
  "X. Dong et al., “Knowledge Vault,” ACM SIGKDD, 2014, pp. 601–610.",
  "D. Krompaß et al., “Type-Constrained Representation Learning in KGs,” arXiv:1508.02593, 2015.",
  "Z. Ji et al., “Survey of Hallucination in NLG,” ACM Comput. Surv., 55(12):1–38, 2023.",
  "A. Bordes et al., “Translating Embeddings (TransE),” NeurIPS, 2013.",
  "B. Yang et al., “Embedding Entities and Relations (DistMult),” arXiv:1412.6575, 2015.",
  "T. Trouillon et al., “Complex Embeddings (ComplEx),” arXiv:1606.06357, 2016.",
  "Z. Sun et al., “RotatE,” arXiv:1902.10197, 2019.",
  "M. Schlichtkrull et al., “Modeling Relational Data with GCNs (R-GCN),” ESWC, 2018.",
  "S. Vashishth et al., “Composition-based Multi-Relational GCNs (CompGCN),” arXiv:1911.03082, 2020.",
  "L. Yao et al., “KG-BERT,” arXiv:1909.03193, 2019.",
  "L. Wang et al., “SimKGC,” arXiv:2203.02167, 2022.",
  "A. Saxena et al., “Sequence-to-Sequence KG Completion (KGT5),” arXiv:2203.10321, 2022.",
  "A. Kochsiek et al., “Friendly Neighbors (KGT5-context),” arXiv:2305.13059, 2023.",
  "L. Yao et al., “Exploring LLMs for KG Completion (KG-LLM),” arXiv:2308.13916, 2025.",
  "P. Rust et al., “Language Modelling with Pixels (PIXEL),” arXiv:2207.06991, 2023.",
  "K. Lee et al., “Pix2Struct,” arXiv:2210.03347, 2023.",
  "A. Karpathy, “The Unreasonable Effectiveness of Visual Representations,” AI Ascent, 2024.",
  "N. Reimers, I. Gurevych, “Sentence-BERT,” arXiv:1908.10084, 2019.",
  "H. Liu et al., “Improved Baselines with Visual Instruction Tuning (LLaVA-1.5),” arXiv:2310.03744, 2024.",
  "T. Dettmers et al., “QLoRA,” arXiv:2305.14314, 2023.",
  "D. Ruffinelli et al., “You CAN Teach an Old Dog New Tricks!,” ICLR, 2020.",
  "J. Devlin et al., “BERT,” arXiv:1810.04805, 2019.",
  "C. Raffel et al., “Exploring the Limits of Transfer Learning (T5),” arXiv:1910.10683, 2020.",
  "H. Liu et al., “Visual Instruction Tuning (LLaVA),” arXiv:2304.08485, 2023.",
  "A. Radford et al., “Learning Transferable Visual Models (CLIP),” arXiv:2103.00020, 2021.",
  "W.-L. Chiang et al., “Vicuna,” 2023.",
  "E. J. Hu et al., “LoRA,” arXiv:2106.09685, 2021.",
  "A. Masry et al., “ChartQA,” arXiv:2203.10244, 2022.",
  "K. Toutanova, D. Chen, “Observed versus latent features (FB15k-237),” CVSC Workshop, 2015, pp. 57–66.",
  "Y. Wei et al., “GITA: Graph to Visual and Textual Integration for Vision-Language Graph Reasoning,” NeurIPS, 2024. arXiv:2402.02130.",
  "Y. Wei et al., “Open the Eyes of MPNN: Vision Enhances MPNN in Link Prediction (GVN/E-GVN),” ICML, 2025. arXiv:2505.08266.",
  "J. Li et al., “ViSR-KGC: Visual Subgraph Reasoning with Vision-Language Models for Multimodal KG Completion,” ACM MM, 2026. arXiv:2608.05833.",
  "B. Fatemi et al., “Talk like a Graph: Encoding Graphs for Large Language Models,” ICLR, 2024. arXiv:2310.04560.",
];
refs.forEach((t, i) => children.push(ref(i + 1, t)));

// ---------- assemble ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: FONT },
        paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("paper/modality_comparison_paper.docx", buf); console.log("written", buf.length, "bytes"); });
