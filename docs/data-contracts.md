# Контракты данных и артефактов

Документ фиксирует форматы артефактов на границе подсистем:

- `simulator` генерирует synthetic-пакет;
- `recognizer` потребляет пакет и публикует результат анализа.

## 1) Версионирование

- Версия формата хранится в каждом `meta.json` как `schema_version`.
- Начальная версия: `starglyph.synthetic.v1`.
- Несовместимые изменения формата => увеличение major (`v2`, `v3`, ...).

## 2) Структура артефактов

```text
artifacts/
  simulator/
    dataset-v1/
      manifest.json
      train/
        frame-000001/
          image.png
          meta.json
          truth-stars.csv
      val/
      test/
  recognizer/
    run-YYYYMMDD-HHMMSS/
      config.json
      summary.json
      per-frame/
        frame-000001.json
      worst-cases/
        frame-000123-overlay.png
```

## 3) Формат synthetic frame package

### 3.0 Величина → интенсивность (рендер)

Пиковая интенсивность в линейных единицах сенсора (до свёртки с PSF и последующего клиппинга) задаётся отображением Погсона относительно опорной звезды. В коде и в этом документе одна и та же формула:

`I_peak = I_ref * 10^(-0.4 * (m - m_ref))`

где `m` — видимая звёздная величина, а `I_ref` и `m_ref` — поля `reference_intensity` и `reference_magnitude` в блоке `render` внутри `meta.json`. Меньшее `m` соответствует более яркой звезде и большему `I_peak` при прочих равных. Итоговые значения пикселей ограничиваются сверху `dynamic_range_max` (реализация: `prototype/crates/simulator-core/src/rendering.rs`).

Каждый пакет `frame-*` содержит минимум:

- `image.png` — синтетический кадр.
- `meta.json` — параметры генерации и камеры.
- `truth-stars.csv` — эталонная разметка **только по звёздам, спроецированным внутрь кадра** (видимым на сенсоре); остальные звёзды каталога в файл не попадают.

### 3.1 `meta.json` (обязательные поля)

```json
{
  "schema_version": "starglyph.synthetic.v1",
  "frame_id": "frame-000001",
  "timestamp_utc": "2026-04-14T21:15:30Z",
  "camera": {
    "width_px": 4032,
    "height_px": 3024,
    "fov_deg": 62.0,
    "intrinsics": {
      "fx": 2850.0,
      "fy": 2850.0,
      "cx": 2016.0,
      "cy": 1512.0
    },
    "distortion": {
      "model": "none",
      "k1": 0.0,
      "k2": 0.0,
      "p1": 0.0,
      "p2": 0.0
    }
  },
  "pose": {
    "ra_deg": 83.633,
    "dec_deg": 22.0145,
    "roll_deg": 5.2
  },
  "render": {
    "psf_sigma_px": 1.2,
    "background_level": 0.06,
    "noise_model": "shot_read"
  },
  "catalog": {
    "name": "hyg-v3",
    "subset": "mag_le_8",
    "license": "CC BY-SA 4.0"
  }
}
```

### 3.2 `truth-stars.csv`

Файл содержит **только** те звёзды, для которых симулятор классифицирует проекцию как **попавшую в прямоугольник кадра** (пиксельные координаты внутри `width_px` × `height_px`). Звёзды вне поля зрения, за плоскостью камеры и т.п. **не перечисляются** — при необходимости полного списка с флагами видимости используйте отдельный derived-артефакт (см. раздел 5).

Колонки (фиксированный порядок):

```text
star_id,ra_deg,dec_deg,mag_v,x_px,y_px,flux_rel,is_in_frame,is_occluded
```

Типы:

- `star_id`: string
- `ra_deg, dec_deg, mag_v, x_px, y_px, flux_rel`: float
- `is_in_frame, is_occluded`: `0|1`

Для формата `starglyph.synthetic.v1` в каждой строке этого файла по определению `is_in_frame = 1` и `is_occluded = 0` (колонки зарезервированы для совместимости схемы и возможных расширений).

## 3.3 Процедура валидации dataset v1 (phase 1 exit gate)

Базовый порядок проверки перед объявлением выхода из фазы 1:

1. Сгенерировать датасет одной командой:
  ```bash
   cd prototype
   cargo run -p dataset-cli -- --seed 42 --output-root artifacts/simulator/dataset-v1 --train-frames 100 --val-frames 20 --test-frames 20 --validate-reproducibility
  ```
2. Запустить регрессионные проверки:
  ```bash
   cd prototype
   make validate-simulator
  ```

Проверка считается успешной, если:

- projection unit tests проходят;
- visual golden проверки не показывают drift;
- reproducibility check подтверждает эквивалентность повторного прогона при том же seed;
- структура артефактов соответствует layout из раздела 2.

## 4) Формат артефактов recognizer

### 4.1 `summary.json`

Содержит агрегированные метрики запуска:

- количество кадров,
- median/p95 по ошибке позы,
- precision/recall детекции,
- latency stats,
- список `worst_cases`.

### 4.2 `per-frame/frame-*.json`

Минимальная структура:

```json
{
  "frame_id": "frame-000001",
  "status": "ok",
  "pose_estimate": {
    "ra_deg": 83.61,
    "dec_deg": 22.03,
    "roll_deg": 5.0
  },
  "errors": {
    "axis_angle_deg": 0.24,
    "roll_deg": 0.20
  },
  "detection": {
    "precision": 0.94,
    "recall": 0.91
  },
  "timings_ms": {
    "detect": 115,
    "match": 172,
    "solve_pose": 43,
    "overlay": 21,
    "total": 351
  }
}
```

## 5) Граница ответственности подсистем

- `simulator` не использует эвристики распознавателя при формировании truth.
- `recognizer` не изменяет synthetic truth, а только читает.
- Любые derived-артефакты (например, "очищенный truth") сохраняются как отдельные файлы с префиксом `derived-`.

