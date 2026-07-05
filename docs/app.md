# Десктоп-приложение Starglyph (архитектура)

Продуктовая цель: GUI-приложение (Rust + Tauri v2), в котором пользователь загружает фото
ночного неба, запускает распознавание (blind plate solving — направление, roll и FOV не
известны заранее) и получает поверх снимка оверлей: линии созвездий, подписи ярких звёзд,
опционально планеты. Обязательный критерий приёмки: система работает на реальных кадрах
`data/input/*.bmp` (аналоговая видеокамера, 740×576, шумные кадры, 5–20 звёзд).

## Состав workspace (`prototype/`)

| Крейт | Роль |
|-------|------|
| `crates/simulator-core`, `crates/dataset-cli` | Фаза 1 — синтетика (без изменений) |
| `crates/solver-core`, `crates/solver-cli` | Фаза 2 — baseline на синтетике (без изменений; референс) |
| `crates/starglyph-core` | **Ядро распознавания реальных фото**: каталог, данные созвездий, загрузка изображений, детекция, blind-солвер, оверлей-геометрия, контракты GUI |
| `crates/starglyph-cli` | Headless-CLI поверх ядра: inspect/detect/solve/batch — приёмка и отладка |
| `apps/desktop` | Tauri-оболочка (`starglyph-desktop`): команды, события прогресса, статический фронтенд `apps/desktop/ui/` |

Принцип из AGENTS.md сохраняется: симулятор и распознаватель не смешиваются;
`starglyph-core` не зависит от `simulator-core` (геометрия реализована заново в тех же
конвенциях и покрыта тестами).

## Модули `starglyph-core`

- `catalog` — HYG v4.2 (`data/catalogs/hyg_v42.csv.gz`, 119 626 звёзд; `rarad/decrad`
  предпочтительнее `ra`(часы!)/`dec`); `Star{id, hip, proper, ra_deg, dec_deg, mag, con}`,
  сортировка по mag, срезы `brighter_than(mag)`.
- `constellations` — d3-celestial GeoJSON (`data/celestial/…`): полилинии в `[ra_deg, dec_deg]`,
  имена по IAU-аббревиатуре.
- `image_input` — BMP/PNG/JPEG → `FrameImage{width, height, gray: Vec<f32> (0..1)}`;
  парсинг таймстампа из имени файла (`YYYY-MM-DD_HH-MM-SS-mmm…`, `CD_YYYY-MM-DD_HHMM`).
- `detect` — фон: вычитание медиан по столбцам/строкам (лечит hot-column x=4 и бандинг),
  mesh-фон 32px с sigma-clip, порог k·σ (MAD), 8-связные компоненты, фильтры
  (границы, area<2 — hot pixels, area>120 — Луна/облака/засветки, elongation — треки),
  субпиксельный центроид по потоку, top-N по flux. Выход: `Detection{x,y,flux,snr,area,rank}`.
- `geom` — единичные векторы RA/Dec, гномоническая проекция/обратная, базис камеры
  (конвенции совпадают с `simulator-core`: экваториальная правая тройка, `x_px = fx·X/Z + cx`,
  `y_px = cy − fy·Y/Z`), кватернионы/матрицы в f64.
- `index` + `solve` — blind-решатель на 4-звёздных геометрических хэшах (семейство
  astrometry.net/tetra3): код инвариантен к подобию → неизвестный FOV решается сопоставлением
  кода, масштаб восстанавливается из отношения угловой/пиксельной диагонали паттерна; базы
  паттернов по полосам FOV (~×1.5: 10–15–23–34–51–76–90°); верификация гипотезы проекцией
  каталога в кадр и вероятностным скорингом; уточнение (кватернион + фокус [+ k1])
  Gauss–Newton/LM по инлаерам; движок сопоставления — крейт `tetra3` (=0.8.0,
  MIT/Apache-2.0, порт ESA tetra3; принят по итогам spike на реальных кадрах), обёрнутый
  нашей независимой верификацией (log-odds по полному списку детекций, приём: гейт tetra3
  **и** ≥6 попаданий, либо log_odds ≥ 18) и LM-уточнением {ra, dec, roll, f, k1}. Базы
  паттернов строятся из HYG при первом запуске (~1 мин, bootstrap 10–70° + плотная полоса
  16–30°) и кэшируются на диск (`prototype/artifacts/cache` у CLI, app-cache у GUI).
- `overlay` — по решённой позе: полилинии созвездий, тесселированные до сегментов ~1°
  и отсечённые кадром, яркие звёзды с подписями (`proper`, иначе Bayer+созвездие), планеты
  (при известной дате), опционально сетка RA/Dec. Собственные движения звёзд применяются
  к эпохе кадра при загрузке каталога (векторный метод по `pmrarad/pmdecrad`); прецессию/
  нутацию/аберрацию НЕ добавлять — их поглощает сама подгонка позы. Проекция ветвится по
  решённому фокусу: гномоническая (FOV ≲ 45°) → + радиальная k1 (до ~70°) → fisheye-модель
  (шире; v2). Всё в пиксельных координатах кадра — фронтенд только рисует.
- `contracts` — сериализуемые DTO для GUI/CLI (см. ниже).
- `ephem` (стретч) — положения планет на дату (выбор реализации по research).

## Контракт GUI (`SolveReport`, JSON)

```json
{
  "status": "solved | failed",
  "failure": {"code": "too_few_stars | no_confident_match | io_error", "message": "…"},
  "pose": {"ra_deg": 0, "dec_deg": 0, "roll_deg": 0},
  "fov": {"fov_x_deg": 0, "fov_y_deg": 0, "focal_px": 0},
  "quality": {"n_detections": 0, "n_inliers": 0, "rms_px": 0, "log_odds": 0, "confidence": 0},
  "timing_ms": {"detect": 0, "solve": 0, "total": 0},
  "detections": [{"x": 0, "y": 0, "flux": 0, "snr": 0, "inlier": true}],
  "overlay": {
    "constellations": [{"abbr": "UMa", "name": "Ursa Major", "lines": [[[0,0],[1,1]]]}],
    "stars": [{"x": 0, "y": 0, "mag": 0, "label": "Vega", "hip": 91262}],
    "planets": [{"x": 0, "y": 0, "name": "Jupiter"}],
    "grid": [{"kind": "ra", "value_deg": 120, "points": [[0,0]]}]
  }
}
```

Точечные координаты — пиксели исходного кадра (f64), фронтенд масштабирует под canvas.

## Tauri-оболочка

- Команды: `load_image(path) → {image_id, width, height, timestamp?}`;
  `solve_image(image_id, on_progress: Channel<Progress>) → SolveReport` (async-команда,
  тяжёлая работа через `spawn_blocking` + rayon внутри);
  `data_attribution() → …` (тексты для About).
- Изображение в webview — через кастомный URI-протокол (`skyimg://frame/<id>?view=raw|stretched`
  → PNG-байты), не base64: нативный кеш webview, без 33% оверхеда.
- Прогресс — `tauri::ipc::Channel` (быстрее и локальнее глобальных событий).
- Состояние: загруженные кадры + лениво построенный индекс (Arc, OnceCell).
- Фронтенд: статические HTML/CSS/JS (без фреймворка и без npm-сборки, `withGlobalTauri`),
  canvas-слои: изображение (raw/stretched переключение — кадры почти чёрные, стретч
  включён по умолчанию), оверлей, слои-переключатели, панель диагностики, зум/пан.

## Данные и лицензии

HYG v4.2 — CC BY-SA 4.0; d3-celestial — BSD-3-Clause (провенанс и чек-суммы:
`THIRD_PARTY_LICENSES.md`, `data/celestial/README.md`). Атрибуция обязана попасть в UI
(About/подвал) до внешнего распространения.

## Сборка и окружение

- Linux: нужны системные `libwebkit2gtk-4.1-dev`, `libgtk-3-dev` и т.п. (Tauri v2).
  В CI/агентной среде без root собирается против user-space sysroot
  (`~/sysroot`, `source ~/sysroot-env.sh`) — см. журнал ветки.
- Проверки: `cargo fmt --check`, `cargo clippy -p starglyph-core -p starglyph-cli -- -D warnings`,
  `cargo test -p starglyph-core -p starglyph-cli`, прогон `starglyph-cli solve` по `data/input/`.

## Приёмка (итоги 2026-07-04)

`cargo run -rq -p starglyph-cli -- batch-solve ../data/input --out-dir artifacts/batch`:

- **решено вслепую 10 из 20 кадров** за ~47 с суммарно (~2.3 с/кадр, 4 CPU-ядра);
  оставшиеся 10 — кадры с 4–6 пригодными звёздами (ниже порога индексируемости паттернов),
  отказ диагностируется («слишком мало звёзд» / «нет уверенного сопоставления»);
- FOV восстанавливается консистентно по всем решениям: 22.2°±0.4° (физический объектив);
- co-pointed пара `CD_0000`/`CD_0020` сходится до 0.07°, серия 2011-09-20 монотонно
  дрейфует по небу (панорамирующая камера) — это свойство данных, не решателя;
- визуальная сверка: «W» Кассиопеи и крест Лебедя ложатся на реальные звёзды кадров,
  подписи (Schedar, Caph, Deneb, Sadr…) на своих местах;
- GUI: решение с прогресс-степпером, оверлей-слои, карточка результата
  (RA/Dec/roll/FOV/инлаеры/rms/уверенность/время), запуск `starglyph-desktop <кадр>
  --auto-solve` для headless-приёмки.

Возможные рычаги для нерешаемых кадров (backlog): поднять предел зв. величины плотной
базы (6.5 → 7.0+), стекинг соседних кадров серии, калибровка дисторсии по уверенным
решениям (`calibrate_camera` tetra3).
