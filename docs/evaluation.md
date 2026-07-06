# Методика оценки качества

Документ задаёт единый способ измерения прогресса распознавателя на synthetic и реальных кадрах.

## 1) Наборы данных для оценки

- `synthetic-clean`: без сложных искажений, контроль геометрии и базового matching.
- `synthetic-realistic`: с шумом/дисторсией/фоном, контроль робастности.
- `real-eval`: небольшой ручной набор реальных фотографий для sanity-check.

Требования к воспроизводимости:

- фиксированный `seed` генератора synthetic;
- зафиксированная версия каталога и формата (`schema_version`);
- сохранение конфигурации запуска recognizer в `config.json`.

## 2) Основные метрики

### 2.1 Ошибка позы

- `axis_angle_deg` — угловое расстояние между true и predicted направлением камеры.
- `roll_error_deg` — модуль разницы true/predicted roll с нормализацией к `[-180, 180]`.

Отчёт:

- `median`, `p95`, `max`.

### 2.2 Качество детекции звёзд

Матч детекции с truth:

- greedy matching по евклидову расстоянию в пикселях;
- порог соответствия: `<= 2.5 px` (для baseline synthetic v1).

Метрики:

- `precision = TP / (TP + FP)`;
- `recall = TP / (TP + FN)`;
- `f1`.

### 2.3 Производительность

Замеряется wall-clock по стадиям:

- `detect_ms`,
- `match_ms`,
- `solve_pose_ms`,
- `overlay_ms`,
- `total_ms`.

Отчёт: `median`, `p95`, `max` по каждому этапу.

## 3) Политика pass/fail для фазы

- Проход фазы определяется порогами из [phase0.md](phase0.md), раздел `0.2`.
- Считаем фазу "условно пройденной", если:
  - все MVP-пороги выполняются на `synthetic-clean`,
  - не менее 80% MVP-порогов выполняются на `synthetic-realistic`,
  - на `real-eval` нет систематического смещения оверлея в одном направлении.

## 4) Формат итогового отчёта запуска

Каждый benchmark-run публикует:

- `summary.json` с агрегированными метриками;
- `per-frame/*.json` для разборов;
- `worst-cases/` с 10 худшими кадрами по `axis_angle_deg`;
- `README.md` (опционально) с краткой интерпретацией результатов.

## 5) Минимальный регламент сравнения версий

- Сравниваются только запуски на одинаковом датасете и одинаковом `config`.
- Улучшение считается значимым, если одновременно:
  - `axis_angle_deg p95` улучшен не менее чем на 10%;
  - `total_ms median` не деградировал более чем на 5%.

## 6) Реальные кадры: харнесс `starglyph eval` (Этап 0 · Epic A)

Харнесс гоняет **живой blind-солвер** (`starglyph-core`) по датасету
[`data/samples/sky-samples/`](../data/samples/sky-samples/README.md) и сравнивает позу с
astrometry.net WCS ground truth (`ground-truth/<id>.wcs.json`). Формат отчёта — §4
(`summary.json`, `per-frame/*.json`, `worst-cases/`), метрики — §2.1/§2.3.

```bash
cd prototype
make eval        # весь track:solver вслепую → artifacts/eval/local/
make eval-gate   # CI-подмножество против prototype/eval/baseline-ci.json
# произвольная конфигурация:
cargo run --release -p starglyph-cli -- eval --manifest ../data/samples/sky-samples/manifest.json \
  --tracks solver,scene --out-dir artifacts/eval/local [--ids a,b] [--baseline <summary.json>]
```

Семантика и решения:

- **Solve-rate считается только по `track:solver`.** `scene` — ожидаемо-нерешаемые
  (сшитые панорамы/пейзаж/композиты; неожиданный solve попадает в
  `scene_track.unexpected_solves` как сигнал); `stress` — только по явному запросу.
- **Замер blind:** кадры решаются независимо, без cross-frame подсказок (двухпроходный
  режим `batch-solve` — продуктовый, не измерительный).
- **Конвенция roll ↔ astrometry `orientation`/`parity`** выведена и зафиксирована в
  `crates/starglyph-core/src/eval.rs` (модульная документация + тесты): при `parity=+1`
  `roll = orientation + 180°` (mod 360); `parity=−1` — зеркальный кадр, физически
  невоспроизводим, roll не сравнивается (`parity_physical=false`). Подтверждено
  эмпирически на `tetra3_alt60` (решён и astrometry, и нашим солвером).
- **Гейт (A3):** `make eval-gate` падает (exit 2) при снижении solve-rate или деградации
  `axis_angle_deg p95` больше чем на 10% (производная §5). Тайминги в гейте не участвуют
  (CI-машины нестабильны по скорости). CI получает кадры по модели «манифест + fetch»:
  лицензионно-чистое подмножество `tetra3_alt40`/`tetra3_alt60` (Apache-2.0) клонируется
  из esa/tetra3 с проверкой sha256, каталог — закоммиченный `hyg_v42.csv.gz`; бинарники
  в git не попадают. Workflow: `.github/workflows/eval-gate.yml`.

Замер на 2026-07-05 (blind, каталог hyg_v3; полный прогон ≈ 2 мин на 8 кадрах):

| Метрика | Значение |
|---|---|
| solve-rate `track:solver` | **1/8 (0.125)** — решён только `tetra3_alt60` (11.4° FOV) |
| Отказы | 7 × `no_confident_match` (в т.ч. `tetra3_alt40`; astrometry решил 7/8) |
| `tetra3_alt60`: axis / roll / fov | 0.008° / 0.60° / 0.4% |
| Тайминг: solve 0.4 с, отказ | медиана 8.3 с, max 15.3 с (deep-retry) |

Разрыв к astrometry (1/8 против 7/8) — **ожидаемый и целевой результат замера**: солвер
заточен под ~22° FOV (`DEFAULT_BLIND_FOV_DEG`), кадры датасета 1°–71°. Закрытие разрыва —
Epic B; харнесс существует, чтобы этот прогресс измерять. Остаточные 0.60° roll на
`tetra3_alt60` при сходимости оси до 29″ — систематическое расхождение
astrometry (SIP-дисторсия) ↔ starglyph (k1-модель), фиксируется честно, не калибруется.
