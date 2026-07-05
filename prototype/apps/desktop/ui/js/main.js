import { hasTauri, imageUrl, invoke, revokeImageUrls, solveImage } from './ipc.js';
import { drawSolveOverlay, ImageViewer } from './viewer.js';

const ALLOWED_EXTENSIONS = new Set(['bmp', 'png', 'jpg', 'jpeg']);

const SOLVE_STAGES = [
  { id: 'load_assets', label: 'Загрузка каталога' },
  { id: 'detect', label: 'Детекция звёзд' },
  { id: 'load_index', label: 'Индекс паттернов' },
  { id: 'match', label: 'Сопоставление' },
  { id: 'verify', label: 'Проверка' },
  { id: 'refine', label: 'Уточнение' },
  { id: 'overlay', label: 'Оверлей' },
];

const STAGE_LABELS = Object.fromEntries(SOLVE_STAGES.map((stage) => [stage.id, stage.label]));

/** @type {{ imageId: number | null, meta: any | null, stretch: boolean, solving: boolean, report: any | null, hasTimestamp: boolean }} */
const appState = {
  imageId: null,
  meta: null,
  stretch: true,
  solving: false,
  report: null,
  hasTimestamp: false,
};

const els = {
  tauriError: document.getElementById('tauri-error'),
  app: document.getElementById('app'),
  btnOpen: document.getElementById('btn-open'),
  btnSolve: document.getElementById('btn-solve'),
  topbarFilename: document.getElementById('topbar-filename'),
  canvasArea: document.getElementById('canvas-area'),
  emptyState: document.getElementById('empty-state'),
  stretchToggle: document.getElementById('stretch-toggle'),
  metaFilename: document.getElementById('meta-filename'),
  metaSize: document.getElementById('meta-size'),
  metaTimestamp: document.getElementById('meta-timestamp'),
  metaExposure: document.getElementById('meta-exposure'),
  metaTimezone: document.getElementById('meta-timezone'),
  solveEmpty: document.getElementById('solve-empty'),
  solveProgress: document.getElementById('solve-progress'),
  solveSteps: document.getElementById('solve-steps'),
  solveResult: document.getElementById('solve-result'),
  solveFailure: document.getElementById('solve-failure'),
  statusHint: document.getElementById('status-hint'),
  statusAttribution: document.getElementById('status-attribution'),
  layerToggles: {
    constellations: document.getElementById('layer-constellations'),
    starNames: document.getElementById('layer-star-names'),
    planets: document.getElementById('layer-planets'),
    detections: document.getElementById('layer-detections'),
    grid: document.getElementById('layer-grid'),
  },
};

const viewer = new ImageViewer(document.getElementById('view'));
viewer.attach(els.canvasArea);
viewer.onAfterDraw = (ctx, transform) => {
  if (!appState.report || appState.report.status !== 'solved') {
    return;
  }
  drawSolveOverlay(
    ctx,
    transform,
    appState.report.overlay,
    appState.report.detections,
    readLayerState(),
  );
};

init();

async function init() {
  if (!hasTauri()) {
    els.tauriError.hidden = false;
    els.app.hidden = true;
    return;
  }

  bindUi();
  await loadAttribution();
  bindDragDrop();
  await handleStartupRequest();
}

function bindUi() {
  els.btnOpen.addEventListener('click', () => openViaDialog());
  els.btnSolve.addEventListener('click', () => runSolve());
  els.stretchToggle.addEventListener('change', () => {
    appState.stretch = els.stretchToggle.checked;
    refreshDisplayedImage();
  });

  for (const input of Object.values(els.layerToggles)) {
    input.addEventListener('change', () => viewer.redraw());
  }

  els.metaTimezone.addEventListener('change', () => {
    void onTimezoneChanged();
  });

  document.getElementById('zoom-out').addEventListener('click', () => viewer.zoomBy(1 / 1.25));
  document.getElementById('zoom-in').addEventListener('click', () => viewer.zoomBy(1.25));
  document.getElementById('zoom-fit').addEventListener('click', () => viewer.fit());
  document.getElementById('zoom-11').addEventListener('click', () => viewer.oneToOne());

  document.addEventListener('keydown', (event) => {
    if (isTypingTarget(event.target)) {
      return;
    }

    if (event.ctrlKey && event.key.toLowerCase() === 'o') {
      event.preventDefault();
      openViaDialog();
      return;
    }

    if (event.key === '+' || event.key === '=') {
      viewer.zoomBy(1.25);
    } else if (event.key === '-') {
      viewer.zoomBy(1 / 1.25);
    } else if (event.key === '0') {
      viewer.fit();
    } else if (event.key === '1') {
      viewer.oneToOne();
    }
  });
}

function bindDragDrop() {
  const tauri = window.__TAURI__;
  const listen = tauri?.event?.listen;
  if (typeof listen !== 'function') {
    return;
  }

  const handleDrop = (event) => {
    const paths = event?.payload?.paths ?? event?.payload;
    if (!Array.isArray(paths) || paths.length === 0) {
      return;
    }
    const firstAllowed = paths.find((path) => isAllowedImagePath(String(path)));
    if (firstAllowed) {
      loadPath(String(firstAllowed));
    } else {
      setStatus('Поддерживаются только BMP, PNG и JPEG.', true);
    }
  };

  listen('tauri://drag-drop', handleDrop).catch(() => {});
  listen('tauri://file-drop', handleDrop).catch(() => {});
}

async function handleStartupRequest() {
  try {
    const request = await invoke('startup_request');
    if (!request?.path) {
      return;
    }
    await loadPath(request.path);
    if (request.auto_solve) {
      await runSolve();
    }
  } catch (error) {
    setStatus(String(error), true);
  }
}

async function loadAttribution() {
  try {
    const info = await invoke('data_attribution');
    const parts = info.items.map((item) => `${item.name} ${item.license}`);
    els.statusAttribution.textContent = `Данные: ${parts.join(' · ')} · v0.1.0`;
  } catch (error) {
    els.statusAttribution.textContent = 'Данные: —';
    setStatus(String(error), true);
  }
}

async function openViaDialog() {
  try {
    const path = await invoke('pick_image');
    if (path) {
      await loadPath(path);
    }
  } catch (error) {
    setStatus(String(error), true);
  }
}

/**
 * @param {string} path
 */
async function loadPath(path) {
  try {
    setStatus('Загрузка снимка…');
    if (appState.imageId !== null) {
      revokeImageUrls(appState.imageId);
    }

    const meta = await invoke('load_image', { path });
    appState.imageId = meta.id;
    appState.meta = meta;
    appState.report = null;
    appState.solving = false;
    appState.hasTimestamp = Boolean(meta.timestamp);

    updateMetaPanel(meta);
    updateTimezoneControl();
    resetSolvePanel();
    updateLayerToggles(false);
    await refreshDisplayedImage();

    els.btnSolve.disabled = false;
    els.btnSolve.classList.remove('is-busy');
    els.emptyState.hidden = true;
    els.topbarFilename.textContent = meta.file_name;
    setStatus('Снимок загружен');
  } catch (error) {
    setStatus(String(error), true);
  }
}

/**
 * @param {any} meta
 */
function updateMetaPanel(meta) {
  els.metaFilename.textContent = meta.file_name;
  els.metaSize.textContent = `${meta.width} × ${meta.height}`;
  els.metaTimestamp.textContent = meta.timestamp ?? '—';
  els.metaExposure.textContent = meta.exposure_label ?? '—';
}

function updateTimezoneControl() {
  els.metaTimezone.disabled = !appState.hasTimestamp;
  if (!appState.hasTimestamp) {
    els.metaTimezone.value = '0';
  }
}

function readUtcOffsetHours() {
  return Number(els.metaTimezone.value);
}

async function onTimezoneChanged() {
  if (
    appState.imageId === null ||
    !appState.hasTimestamp ||
    !appState.report ||
    appState.report.status !== 'solved'
  ) {
    return;
  }

  try {
    setStatus('Пересчёт планет…');
    const overlay = await invoke('recompute_overlay', {
      id: appState.imageId,
      utcOffsetHours: readUtcOffsetHours(),
    });
    appState.report = { ...appState.report, overlay };
    viewer.redraw();
    setStatus('Планеты обновлены');
  } catch (error) {
    setStatus(String(error), true);
  }
}

async function refreshDisplayedImage() {
  if (appState.imageId === null) {
    viewer.clearImage();
    els.emptyState.hidden = false;
    return;
  }

  const view = appState.stretch ? 'stretched' : 'raw';
  const url = await imageUrl(appState.imageId, view);
  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error('не удалось декодировать PNG снимка'));
    img.src = url;
  });
  viewer.setImage(img);
}

function resetSolvePanel() {
  els.solveEmpty.hidden = false;
  els.solveProgress.hidden = true;
  els.solveResult.hidden = true;
  els.solveFailure.hidden = true;
  els.solveSteps.replaceChildren();
}

async function runSolve() {
  if (appState.imageId === null || appState.solving) {
    return;
  }

  appState.solving = true;
  els.btnSolve.disabled = true;
  els.btnSolve.classList.add('is-busy');
  els.solveEmpty.hidden = true;
  els.solveResult.hidden = true;
  els.solveFailure.hidden = true;
  els.solveProgress.hidden = false;

  const completedStages = new Set();
  /** @type {string | null} */
  let activeStage = null;
  renderSolveSteps(completedStages, activeStage);
  setStatus('Распознавание…');

  try {
    const report = await solveImage(appState.imageId, readUtcOffsetHours(), (event) => {
      const stage = mapProgressStage(event?.stage);
      if (!stage) {
        return;
      }
      if (activeStage) {
        completedStages.add(activeStage);
      }
      activeStage = stage;
      renderSolveSteps(completedStages, activeStage);
    });

    if (activeStage) {
      completedStages.add(activeStage);
    }
    renderSolveSteps(completedStages, null);

    appState.report = report;

    if (report.status === 'solved') {
      renderSolveResult(report);
      updateLayerToggles(true);
      const pose = report.pose ?? {};
      setStatus(`Решено: RA ${formatRaHms(pose.ra_deg)} Dec ${formatDecDms(pose.dec_deg)}`);
    } else {
      renderSolveFailure(report);
      updateLayerToggles(false);
      setStatus('Не решено', true);
    }
    viewer.redraw();
  } catch (error) {
    renderSolveFailure({ failure: { code: 'io_error', message: String(error) } });
    setStatus(String(error), true);
  } finally {
    appState.solving = false;
    els.btnSolve.disabled = appState.imageId === null;
    els.btnSolve.classList.remove('is-busy');
  }
}

/**
 * @param {Set<string>} completedStages
 * @param {string | null} activeStage
 */
function renderSolveSteps(completedStages, activeStage) {
  els.solveSteps.replaceChildren();

  for (const stage of SOLVE_STAGES) {
    const li = document.createElement('li');
    li.className = 'solve-step';

    const icon = document.createElement('span');
    icon.className = 'step-icon';

    if (completedStages.has(stage.id)) {
      li.classList.add('is-done');
      icon.textContent = '✓';
    } else if (stage.id === activeStage) {
      li.classList.add('is-active');
      const spinner = document.createElement('span');
      spinner.className = 'spinner';
      spinner.setAttribute('aria-hidden', 'true');
      icon.appendChild(spinner);
    } else {
      icon.textContent = '○';
    }

    const label = document.createElement('span');
    label.textContent = stage.label;

    li.append(icon, label);
    els.solveSteps.appendChild(li);
  }
}

/**
 * @param {any} report
 */
function renderSolveResult(report) {
  els.solveProgress.hidden = true;
  els.solveResult.hidden = false;

  const pose = report.pose ?? {};
  const fov = report.fov ?? {};
  const quality = report.quality ?? {};
  const timing = report.timing_ms ?? {};

  els.solveResult.innerHTML = '';
  const title = document.createElement('strong');
  title.textContent = 'Решение найдено';
  els.solveResult.appendChild(title);

  const dl = document.createElement('dl');
  dl.className = 'result-grid tabular-nums';
  appendResultRow(
    dl,
    'Центр',
    `RA ${formatRaHms(pose.ra_deg)} / Dec ${formatDecDms(pose.dec_deg)}`,
  );
  appendResultRow(dl, 'Roll', formatAngleOne(pose.roll_deg));
  appendResultRow(
    dl,
    'Поле зрения',
    `${formatAngleOne(fov.fov_x_deg)} × ${formatAngleOne(fov.fov_y_deg)}`,
  );
  appendResultRow(
    dl,
    'Звёзд сопоставлено',
    `${quality.n_inliers ?? '—'} / ${quality.n_detections ?? '—'}`,
  );
  appendResultRow(dl, 'Точность', formatPx(quality.rms_px));
  appendResultRow(dl, 'Уверенность', formatConfidence(quality.log_odds));
  appendResultRow(
    dl,
    'Время',
    `${timing.total ?? '—'} мс (${timing.detect ?? '—'}+${timing.solve ?? '—'})`,
  );
  els.solveResult.appendChild(dl);
}

/**
 * @param {any} report
 */
function renderSolveFailure(report) {
  els.solveProgress.hidden = true;
  els.solveFailure.hidden = false;
  const failure = report.failure ?? {};
  els.solveFailure.textContent = mapFailureMessage(
    failure.code,
    failure.message,
    report.detections,
  );
}

/**
 * @param {boolean} solved
 */
function updateLayerToggles(solved) {
  const defaults = {
    constellations: true,
    starNames: true,
    planets: true,
    detections: false,
    grid: false,
  };

  for (const [key, input] of Object.entries(els.layerToggles)) {
    if (key === 'planets') {
      const planetsAvailable = solved && appState.hasTimestamp;
      input.disabled = !solved || !appState.hasTimestamp;
      input.title = planetsAvailable ? '' : solved ? '' : 'Появится после распознавания';
      if (!planetsAvailable) {
        input.title = solved ? 'Нет даты снимка' : 'Появится после распознавания';
      }
      input.checked = planetsAvailable ? defaults.planets : false;
      continue;
    }

    input.disabled = !solved;
    input.title = solved ? '' : 'Появится после распознавания';
    if (solved) {
      input.checked = defaults[key] ?? false;
    } else {
      input.checked = false;
    }
  }
}

function readLayerState() {
  return {
    constellations: els.layerToggles.constellations.checked,
    starNames: els.layerToggles.starNames.checked,
    planets: els.layerToggles.planets.checked,
    detections: els.layerToggles.detections.checked,
    grid: els.layerToggles.grid.checked,
  };
}

/**
 * @param {string | undefined} stage
 */
function mapProgressStage(stage) {
  if (!stage) {
    return null;
  }
  const normalized = stage.toLowerCase();
  if (STAGE_LABELS[normalized]) {
    return normalized;
  }
  if (normalized.includes('detect')) return 'detect';
  if (normalized.includes('index')) return 'load_index';
  if (normalized.includes('match')) return 'match';
  if (normalized.includes('verify')) return 'verify';
  if (normalized.includes('refine')) return 'refine';
  if (normalized.includes('overlay')) return 'overlay';
  if (normalized.includes('asset') || normalized.includes('catalog')) return 'load_assets';
  return normalized;
}

/**
 * @param {string | undefined} code
 * @param {string | undefined} message
 * @param {any[] | undefined} detections
 */
function mapFailureMessage(code, message, detections) {
  switch (code) {
    case 'not_implemented':
      return 'Решатель ещё не подключён (следующий этап разработки)';
    case 'too_few_stars': {
      const count = detections?.length ?? parseStarCount(message);
      return `На снимке слишком мало звёзд (найдено ${count ?? '?'})`;
    }
    case 'no_confident_match':
      return 'Не удалось уверенно сопоставить участок неба';
    case 'io_error':
      return message || 'Ошибка ввода-вывода';
    default:
      return message || 'Неизвестная ошибка распознавания';
  }
}

/**
 * @param {string | undefined} message
 */
function parseStarCount(message) {
  if (!message) {
    return null;
  }
  const match = message.match(/(\d+)\s+star/i);
  return match ? Number(match[1]) : null;
}

/**
 * @param {string} path
 */
function isAllowedImagePath(path) {
  const ext = path.split('.').pop()?.toLowerCase() ?? '';
  return ALLOWED_EXTENSIONS.has(ext);
}

/**
 * @param {EventTarget | null} target
 */
function isTypingTarget(target) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || target.isContentEditable;
}

/**
 * @param {string} text
 * @param {boolean} [isError]
 */
function setStatus(text, isError = false) {
  els.statusHint.textContent = text;
  els.statusHint.classList.toggle('is-error', isError);
}

/**
 * @param {HTMLDListElement} dl
 * @param {string} label
 * @param {string} value
 */
function appendResultRow(dl, label, value) {
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = value;
  dl.append(dt, dd);
}

/**
 * @param {number | undefined} deg
 */
function formatRaHms(deg) {
  if (typeof deg !== 'number' || !Number.isFinite(deg)) {
    return '—';
  }
  let hours = (deg / 15) % 24;
  if (hours < 0) {
    hours += 24;
  }
  const h = Math.floor(hours);
  const minutesTotal = (hours - h) * 60;
  const m = Math.floor(minutesTotal);
  const s = Math.round((minutesTotal - m) * 60);
  return `${h}h ${m}m ${s}s`;
}

/**
 * @param {number | undefined} deg
 */
function formatDecDms(deg) {
  if (typeof deg !== 'number' || !Number.isFinite(deg)) {
    return '—';
  }
  const sign = deg < 0 ? '−' : '+';
  const abs = Math.abs(deg);
  const d = Math.floor(abs);
  const minutesTotal = (abs - d) * 60;
  const m = Math.floor(minutesTotal);
  const s = Math.round((minutesTotal - m) * 60);
  return `${sign}${d}° ${m}′ ${s}″`;
}

/**
 * @param {number | undefined} value
 */
function formatAngleOne(value) {
  return typeof value === 'number' ? `${value.toFixed(1)}°` : '—';
}

/**
 * @param {number | undefined} value
 */
function formatPx(value) {
  return typeof value === 'number' ? `${value.toFixed(2)} px` : '—';
}

/**
 * @param {number | undefined} logOdds
 */
function formatConfidence(logOdds) {
  if (typeof logOdds !== 'number') {
    return '—';
  }
  if (logOdds >= 30) {
    return 'высокая';
  }
  if (logOdds >= 18) {
    return 'средняя';
  }
  return 'низкая';
}
