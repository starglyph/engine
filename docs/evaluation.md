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
  `crates/starglyph-core/src/eval.rs` (модульная документация + тесты). Она **зависит от
  контейнера исходника** (nova по-разному переворачивает строки при конвертации):
  16-бит TIFF → `roll = orientation + 180°`; consumer JPEG → `roll = −orientation`
  (вертикальный флип, θ→180−θ). Выбор — `WcsRowConvention::for_image_path`;
  откалибровано по 5 кадрам с inlier-подтверждёнными позами живого солвера
  (расхождения ≤ 2.8°). `parity=−1` — зеркальный кадр, физически невоспроизводим,
  roll не сравнивается (`parity_physical=false`).
- **Гейт (A3):** `make eval-gate` падает (exit 2) при снижении solve-rate или деградации
  `axis_angle_deg p95` больше чем на 10% (производная §5). Тайминги в гейте не участвуют
  (CI-машины нестабильны по скорости). CI получает кадры по модели «манифест + fetch»:
  лицензионно-чистое подмножество `tetra3_alt40`/`tetra3_alt60` (Apache-2.0) клонируется
  из esa/tetra3 с проверкой sha256, каталог — закоммиченный `hyg_v42.csv.gz`; бинарники
  в git не попадают. Workflow: `.github/workflows/eval-gate.yml`.

Замер на 2026-07-05 (blind, каталог hyg_v3; **до** Epic B):

| Метрика | Значение |
|---|---|
| solve-rate `track:solver` | **1/8 (0.125)** — решён только `tetra3_alt60` (11.4° FOV) |
| Отказы | 7 × `no_confident_match` (в т.ч. `tetra3_alt40`; astrometry решил 7/8) |
| `tetra3_alt60`: axis / roll / fov | 0.008° / 0.60° / 0.4% |
| Тайминг: solve 0.4 с, отказ | медиана 8.3 с, max 15.3 с (deep-retry) |

Замер на 2026-07-06 (blind, каталог hyg_v3; **после** B1+B2+B3-среза: EXIF-prior,
слепая лестница dense-бандов 22°/40°/65°, адаптивные пороги детекции):

| Метрика | Значение |
|---|---|
| solve-rate `track:solver` | **5/8 (0.625)**: + `tetra3_alt40` (11.4°), `wm_constellation_orion` (27.5°), `flickr_cygnus_fermion` (36.3°), `flickr_orion_rahn` (71.2°) |
| Отказы (3) | `flickr_torchbearer_ladia` (astrometry тоже не решил), `flickr_m41_donatiello` (1.0° — за пределами mag 6.5 базы, → B4), `eso_cerro_armazones` (3.3°, parity=−1 зеркальный композит) |
| axis_angle_deg (5 GT-кадров) | median 0.067° · p95 0.53° · max 0.64° (`orion_rahn`, широкая линза без модели дисторсии → B5) |
| roll_error_deg | median 0.49° · max 2.81° (SIP ↔ k1, честный резидуал) |
| fov_error_rel | median 1.3% · max 9.6% (`orion_rahn`, дисторсия → B5) |
| Тайминг total | median 3.8 c · p95 15.8 c (отказы платят полную лестницу бандов) |

Разрыв к astrometry сокращён с 1/8↔7/8 до 5/8↔7/8; оставшиеся два кадра — узкое поле
(≤3.3°), требующее более глубокой mag-базы (Epic B4), и зеркальный композит (не цель
живого солвера). Широкоугольный телефонный диапазон (~60–75°) впервые решается вслепую.
