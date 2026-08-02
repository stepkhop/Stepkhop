# Степан Харитонов

## ML Engineer | Deep Learning | Scientific Computing

Занимаюсь разработкой ML/AI-систем на стыке deep learning и научных вычислений: от оптимизации нейросетей в условиях жёстких вычислительных ограничений до построения фреймворков для оценки качества RAG-пайплайнов.

---

## Deep Learning Research — SMILE: Zero-Order Optimization of ResNet18 under Compute Constraints

**Задача.** Обучение глубокой нейросети (ResNet18) в условиях, когда аналитические градиенты недоступны, а вычислительный бюджет жёстко ограничен (≤ 256 forward passes).

**Подход.**
- **Zero-Order Optimization (SPSA)** — градиент аппроксимируется через симметричные возмущения параметров без backpropagation:

$$
g(\theta) \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}
$$

- **Гибридная схема оптимизации** — SPSA для исследования пространства параметров в связке с Adam для стабилизации обновлений.
- **Стратегия обучения** — curriculum learning, поэтапная разморозка слоёв (staged unfreezing), стабилизация в low-compute режиме.

**Результат.** Accuracy выросла с 4.12% до 21.47% при сохранении жёсткого ограничения на число forward-pass операций и стабильной сходимости.

**Стек:** Python, PyTorch, NumPy

🔗 Репозиторий: [Solving-the-problem-on-SMILE](https://github.com/stepkhop/Solving-the-problem-on-SMILE)

---

## RAGNAROK — RAG Numeric Analytics, Root-cause & Outcome Kit

Модульный, независимый от LLM фреймворк для комплексной оценки RAG-пайплайнов (Retrieval-Augmented Generation), разработанный в рамках проекта **SMILES-2026**.

**Что делает фреймворк:**
- Оценивает retrieval, генерацию и бизнес-эффект RAG-пайплайна и сводит результат к единому **Business Score (0–100)**.
- Автоматически диагностирует, **на каком этапе** пайплайна и **по какой причине** возникает проблема (чанкинг, эмбеддинги, реранкинг, генерация, бизнес-ограничения).
- Не требует обращения к LLM для базовых метрик — работает на локальных embedding/NLI-моделях; LLM-судья доступен как опциональное дополнение.

**Ключевые возможности:**
- 5-этапная диагностика retrieval-пайплайна с автоматическим определением узкого места (bottleneck): доступность данных → качество чанкинга → качество эмбеддингов → качество ретрива (Hit Rate@k, Precision@k, Recall@k, NDCG@k, MRR@k) → эффект реранкинга.
- Метрики генерации: faithfulness (NLI-энтейлмент, детект галлюцинаций), Semantic Footprint Coverage (детект пропусков), Information Density Score (детект избыточности).
- Бизнес-метрики: стоимость (токены / GPU-часы), латентность, доверие (trust), эффективность покрытия контекста, композитный Business Score с настраиваемыми весами.
- Готовые пресеты конфигурации (safety-first, speed-first, cost-saving, medical, customer support, research и др.), выбор по названию, свободному тексту или числовым приоритетам.
- Структурированные отчёты (Markdown/HTML/dict) без обращения к LLM, либо LLM-нарратив как опция.
- Покрытие тестами (pytest), YAML-конфигурация, синтетический набор из 30 размеченных тест-кейсов для проверки классификатора ошибок.

**Стек:** Python

🔗 Репозиторий: [RAGNAROK](https://github.com/aameliig/RAGNAROK)

---

## Скоро

- Научная публикация по математическому моделированию (transport of substances / respiratory mechanics в лёгких) — раздел будет дополнен.

---

## Навыки

Python · PyTorch · NumPy · Deep Learning · Zero-Order Optimization · RAG Evaluation · Scientific Computing · Mathematical Modeling · Git

## Контакты

- Telegram: [@kra23z](https://t.me/kra23z)
- GitHub: [stepkhop](https://github.com/stepkhop)
- Email: stepan2007p@mail.ru
