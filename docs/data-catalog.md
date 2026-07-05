# Каталог источников реальных снимков (проверенный)

Результат разведки источников широкоугольных снимков ночного неба для
blind plate solving. Собран параллельным поиском по пяти классам источников
(Astrometry.net/WCS, Wikimedia Commons, Flickr/Openverse/сток-платформы,
институциональные архивы, академические датасеты) с **поштучной проверкой
лицензий**. Дата разведки: **2026-07-05**.

> Политику, риски и шаблон согласия см. в [data-sources.md](data-sources.md).
> Собранная на основе каталога выборка (манифест + скрипт пересборки, **без
> бинарников**) — в [`../data/samples/sky-samples/`](../data/samples/sky-samples/).

## Как читать таблицы

Источники разложены по **трём тирам совместимости** с публичным репозиторием
под Apache-2.0:

- 🟢 **Тир A — безопасно.** Пермиссивные лицензии без share-alike:
  CC0, Public Domain, CC BY 4.0/3.0/2.0, Apache-2.0, BSD-3. Требуют максимум
  атрибуции. Медиа держим под их собственной лицензией (не наследуют Apache-2.0 кода).
- 🟡 **Тир B — с трением.** Copyleft/ShareAlike (CC BY-SA, CC BY-SA IGO):
  хранить/использовать можно с атрибуцией, но **любой производный** кадр (кроп,
  денойз, аннотация) при перераспространении обязан остаться под BY-SA. Смешанная
  лицензия. Предпочтительно ссылаться, а не вендорить.
- 🔴 **Тир C — не использовать.** Кастомные сток-лицензии, копирайт фотографа,
  NC-варианты, неуказанная лицензия.

> Мы применяем модель **манифест + fetch**: в репозиторий попадают только ссылки,
> лицензии и контрольные суммы, а не сами изображения. Это снимает вопрос о
> перераспространении даже для Тир B и держит публичный репо лёгким.

---

## 🟢 Тир A — безопасно (пермиссивные)

| Источник | Что это | Лицензия | Атрибуция | Широкое поле? | Ground truth (WCS)? | Уверенность |
|---|---|---|---|---|---|---|
| **ESA tetra3** — [github.com/esa/tetra3](https://github.com/esa/tetra3) | Реальные кадры FLIR Blackfly + 35 мм, ~11.4° FOV, в `examples/data/` | **Apache-2.0** (весь репозиторий, LICENSE.txt) | © ESA + NOTICE | Да (узкое) | Частично (решаются солвером; полнота WCS-меток не подтверждена) | Высокая — **чистейший вариант** ✅ в выборке |
| **cedar-solve** — [github.com/smroid/cedar-solve](https://github.com/smroid/cedar-solve) | Форк tetra3, реф-солвер + те же тестовые кадры | **Apache-2.0** | upstream tetra3/ESA + автор форка | Да | Как tetra3 | Средне-высокая |
| **NASA COTS-Star-Tracker** — [github.com/nasa/COTS-Star-Tracker](https://github.com/nasa/COTS-Star-Tracker) | Есть `data/`+`examples/`, выдаёт кватернион позы | **BSD-3-Clause** (код); лицензия данных не указана | © соответственно | ? (проверить) | Возможно; проверить наличие меток | Низкая по данным — **инспектировать перед использованием** |
| **ESO** — [eso.org/public/images](https://www.eso.org/public/images/) | GigaGalaxy Zoom, VISTA-мозаики, all-sky панорамы | **CC BY 4.0** | «ESO/S. Brunier» и т.п. (verbatim) | **Да, отлично** | Нет (но известны координаты поля) | Высокая — **основной институциональный источник** ✅ в выборке |
| **NOIRLab / NSF** — [noirlab.edu/public/images](https://noirlab.edu/public/images/) | Image of the Week: Млечный Путь над телескопами | **CC BY 4.0** | Многосоставная строка, копировать точно с каждой страницы | **Да, отлично** | Нет | Высокая (лицензия); строку credit брать со страницы ✅ в выборке |
| **ESA/Hubble, ESA/Webb** — [esahubble.org](https://esahubble.org/copyright/) | В осн. deep-space, редкие широкие кадры | **CC BY 4.0** | «ESA/Hubble» (verbatim) | Ограниченно | Нет | Высокая |
| **nova.astrometry.net** (анонимные загрузки) — [nova.astrometry.net/user_images](https://nova.astrometry.net/user_images) | Любительские кадры с **решённой астрометрией** | **CC BY 3.0** (default для анонимных) | «Image by <user>, via nova.astrometry.net, CC BY 3.0» | Да | **Да — `/api/jobs/<jobid>/calibration/` (без авторизации), `wcs_file/<JOBID>`** | ⚠ **поштучно** — авторизованные юзеры могут ставить NC/SA; связка кадр→job→лицензия рендерится через JS |
| **Flickr — CC0 (id 9) / PDM (id 10)** — [фильтр 9,10](https://www.flickr.com/search/?text=night+sky+stars&license=9,10) | Любительские широкоугольные кадры | **CC0 / Public Domain** | Не требуется (courtesy) | Да | Нет | Высокая по модели; **поштучно** ✅ в выборке |
| **Flickr — CC BY / CC BY-SA (4,5,11,12)** | То же | CC BY / CC BY-SA 2.0/4.0 | Требуется | Да | Нет | Высокая по модели; поштучно |
| **Wikimedia Commons — CC0 / CC BY items** — [Featured pictures/Astronomy](https://commons.wikimedia.org/wiki/Commons:Featured_pictures/Astronomy) | Отфильтрованные широкие кадры | CC0 / CC BY 2.0/3.0/4.0 (в категориях есть и BY-SA — см. Тир B) | По `AttributionRequired` из API | Да | Нет | Средняя — **поштучно через API** ✅ в выборке |
| **DeepSpaceYoloDataset** — [Zenodo 8387071](https://doi.org/10.5281/zenodo.8387071) | Реальные астрофото smart-телескопа | **CC BY 4.0** | Цитата MDPI Data 2024 + DOI | Умеренно (deep-sky) | GT = bbox объектов, **не WCS** | Высокая (лицензия) — предпочесть ссылку, не вендор |

## 🟡 Тир B — ShareAlike-трение (copyleft)

| Источник | Что это | Лицензия | Почему трение | Ground truth? |
|---|---|---|---|---|
| **Wikimedia Commons — основная масса** — [Category:Night_sky](https://commons.wikimedia.org/wiki/Category:Night_sky), [Milky_Way…place_on_Earth](https://commons.wikimedia.org/wiki/Category:Milky_Way_Galaxy_and_a_place_on_Earth), [Star_trails](https://commons.wikimedia.org/wiki/Category:Star_trails) | Любительские широкие кадры МП, star-trails | **CC BY-SA 4.0/3.0** | Производные (кроп/денойз/аннотация) при раздаче обязаны остаться BY-SA | Нет |
| **ESA/Gaia all-sky maps** — [sci.esa.int/gaia](https://sci.esa.int/web/gaia/-/60169-gaia-s-sky-in-colour) | Полнонебесные карты | **CC BY-SA 3.0 IGO** | ShareAlike IGO; «ESA/Gaia/DPAC» | Всё небо по определению |
| **ESA StarNav / MOON15** — [Zenodo 15166001](https://doi.org/10.5281/zenodo.15166001) | Захват сенсора star-tracker | **CC BY-SA 4.0** | ShareAlike + это **кадры Луны**, не звёздные поля | Калибровка; поза не подтверждена |

## 🔴 Тир C — не использовать

| Источник | Причина запрета |
|---|---|
| **Unsplash / Pexels / Pixabay (после 2019)** | Кастомные лицензии, **не** Creative Commons. Прямо запрещают перераспространение файлов как standalone/датасет; Unsplash+Pixabay также запрещают ML-обучение. |
| **APOD (apod.nasa.gov)** | Несмотря на домен nasa.gov, большинство снимков — **© конкретного астрофотографа**. Не PD. |
| **astrometry.net `demo/apod*.jpg`** | Реальные APOD-фото, **индивидуально копирайтные** (см. `demo/CREDITS`). Код репо ≈ GPLv3 (вызывать как внешний инструмент, не встраивать). |
| **esa.int / sci.esa.int (default)** | Дефолтные условия ESA — образовательно-редакционные, **не** CC, ограничивают коммерческое использование. Только явно помеченные CC BY-SA 3.0 IGO (→ Тир B). |
| **nova.astrometry.net — NC-варианты** | Авторизованные загрузки могут быть CC BY-NC / BY-NC-SA → несовместимо с пермиссивным датасетом. |
| **Flickr NC / ND (id 1,2,3,6,13,14,15,16)** | NonCommercial / NoDerivatives конфликтуют с открытым переиспользуемым датасетом. |
| **EBS-EKF (Kitware)** | Лучший GT (синхронная поза star-tracker), но **лицензия не указана** + event-camera поток. Citation ≠ license. Писать авторам. |
| **Kaggle-датасеты (SDSS17, star-type…)** | Лицензии не указаны/специфичны, данные классификационные, не звёздные поля/WCS. |
| **IEEE DataPort «Star Sensor Image»** | Подписочный доступ; синтетика. |

---

## Ground truth: где реально есть привязка

Для оценки точности солвера нужны кадры с известными RA/Dec/FOV/roll. Подробно —
[`../data/samples/sky-samples/GROUND-TRUTH.md`](../data/samples/sky-samples/GROUND-TRUTH.md).

1. **nova.astrometry.net** — `/api/jobs/<jobid>/calibration/` отдаёт полный WCS
   **без авторизации** (проверено); но кадр→job→лицензия рендерится через JS, поэтому
   массовый лицензионно-чистый харвест хрупок.
2. **Свои кадры → astrometry.net API** — [`solve_wcs.py`](../data/samples/sky-samples/solve_wcs.py):
   решаем уже-чистые CC0/Apache/CC-BY кадры, получаем WCS (координаты — факты, не объект
   авторского права). Чистый путь.
3. **Собственный blind-солвер движка** — канонический источник GT (без внешних лимитов).
4. **Симулятор проекта** — синтетика с точным `meta.json`.

## Сквозные правила

1. **EXIF/PII.** Ни Commons, ни Flickr, ни любители EXIF не чистят (встречаются GPS +
   таймстампы = ПДн). Пересборщик [`fetch_sample.py`](../data/samples/sky-samples/fetch_sample.py)
   вычищает EXIF при каждом восстановлении.
2. **Лицензию — с первоисточника, поштучно.** Openverse, Kaggle, миниатюры — не авторитет.
   Контрпример: flickr.com/photos/ironrodart/6052371623 выглядит идеально, но All Rights Reserved.
3. **Медиа ≠ код.** Файлы не наследуют Apache-2.0. Не-CC0 кадр → строка в
   [ATTRIBUTION](../data/samples/sky-samples/ATTRIBUTION.md) с точным текстом и URL лицензии.
4. **Предпочитать CC0 / CC BY 4.0 / PD** ради избежания ShareAlike-трения.
5. **Citation ≠ license.** BibTeX/DOI прав на распространение не даёт.
6. **Код-лицензия ≠ дата-лицензия** (ловушка astrometry.net demo).
7. **Модель манифест + fetch** — не храним чужие бинарники в репо; храним провенанс.

## Собранная выборка

Реализована в [`../data/samples/sky-samples/`](../data/samples/sky-samples/) — **23 кадра**
(21 Тир-A + 2 Тир-B), лицензии проверены поштучно, провенанс в `manifest.json`
(page_url, download_url, sha256, license_url на ассет), пересборка — `fetch_sample.py`:

- **Реальные сенсорные / узкое поле** (Apache-2.0): 2× tetra3 (~11° FOV, 16-бит).
- **Любительские широкоугольные / созвездия** (CC0): Orion, Torch Bearer, Cygnus, Constellation Orion.
- **Плотное поле / скопление** (CC0): M41.
- **Институциональные широкоугольные / all-sky** (CC BY 4.0): ESO панорама + VISTA, Milky Way Arch, 2× NOIRLab.
- **Широкоугольные, разные условия** (CC BY 2.0/3.0/4.0): комета NEOWISE, шумный любительский, Гималаи, портрет, центр МП, дуга МП, ESO Armazones, Сочи (RU), панорама.
- **🟡 Тир-B стресс** (CC BY-SA 4.0): звёздные треки, след спутника.

Следующий проход: WCS-сайдкары (`solve_wcs.py`), больше стресса (засветка/градиент,
дисторсия, ночной режим), расширение Тир-A (в пуле Wikimedia ещё ~44 PD + десятки CC0/CC BY).
