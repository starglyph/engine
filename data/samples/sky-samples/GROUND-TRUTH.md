# Ground truth (WCS/поза) для выборки — статус и путь

Для оценки точности солвера нужны кадры с известными **RA/Dec центра, FOV/pixscale,
ориентацией (roll), parity**. В текущей выборке готового WCS почти нет: tetra3-кадры
решаются солвером, остальное — снимки без привязки. Ниже — что выяснено и как получать GT.

## Что проверено (2026-07-05)

- **`nova.astrometry.net/api/jobs/<jobid>/calibration/` отдаёт полный WCS без авторизации.**
  Пример (job 1): `{"ra":169.807, "dec":3.426, "pixscale":0.198, "orientation":-0.104, "parity":-1.0, "radius":0.127}`.
  Также `/api/jobs/<jobid>/info/` — статус, `original_filename`, calibration.
- **`nova.astrometry.net/wcs_file/<jobid>`** отдаёт FITS-заголовок с WCS (`application/fits`).
- **НО:** страница `user_images/<id>` (кадр → job id → **лицензия**) рендерится через
  JavaScript; в статическом HTML их нет. Значит массовый «скачать чужие решённые кадры
  с проверкой лицензии» — хрупок и **лицензионно рискован** (по кадру лицензия может быть
  CC BY-NC / BY-SA, а не пермиссивная). Не делаем.

## Правильный путь: решать СВОИ уже-чистые кадры

Лицензия у наших CC0/Apache/CC-BY кадров уже чистая; вычисленный WCS — это **факты**
(небесные координаты), не объект авторского права. Поэтому:

```bash
export ASTROMETRY_API_KEY=xxxx          # бесплатно: https://nova.astrometry.net/api_help
python3 solve_wcs.py images/B_amateur_widefield_cc0__flickr_orion_rahn.jpg images/*.jpg
```

На выходе рядом с кадром: `<frame>.wcs.json` (ra/dec/pixscale/orientation/parity +
объекты в поле) и `<frame>.wcs.fits` (полный WCS-заголовок). Это и есть ground truth
для приёмочного/регрессионного набора.

Кандидаты на первый прогон (чистые точечные звёзды, должны решиться уверенно):
tetra3 (уже узкое поле), `flickr_orion_rahn`, `flickr_cygnus_fermion`,
`wm_constellation_orion`, `flickr_m41_donatiello`. Обработанные композиты ESO/NOIRLab и
кадры с сильной засветкой/треками — как проверка устойчивости (могут не решиться — это
тоже полезный сигнал).

## Канонический источник GT в проекте

Внешний astrometry.net — bootstrap и кросс-чек. **Основной** ground truth должен давать
**собственный blind-солвер движка** (без внешних зависимостей и лимитов) — см. методику
в [`../../../docs/evaluation.md`](../../../docs/evaluation.md). Плюс **синтетика симулятора**
с точным `meta.json` — для кривых обучения и регрессий, где истина известна по построению.

## Готовые сайдкары

`ground-truth/tetra3_alt60.wcs.json`, `ground-truth/tetra3_alt40.wcs.json` — получены
bootstrap-прогоном `solve_wcs.py` через astrometry.net API (2026-07-05) на Apache-2.0
кадрах tetra3. Содержат только факты: центр RA/Dec, pixscale, orientation, parity,
объекты в поле (поле ~11.5° в Corona Borealis). Отправка была приватной
(`publicly_visible=n`). Остальные кадры пока без WCS — досоответить локальным солвером,
без внешних загрузок.

## Итого

| Источник GT | Доступ | Плюсы | Минусы |
|---|---|---|---|
| astrometry.net API (`solve_wcs.py`) | нужен бесплатный ключ | реальные кадры, точный WCS | внешний сервис, лимиты, async |
| nova calibration API (read) | без ключа | быстрый read по job id | нужен job id + лицензия кадра (JS) |
| Собственный солвер движка | локально | канон, без лимитов, воспроизводимо | требует готового eval-harness (Epic A) |
| Симулятор проекта | локально | истина по построению | синтетика, не реальные артефакты |
