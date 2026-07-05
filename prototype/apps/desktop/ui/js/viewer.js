/** @typedef {{ scale: number, ox: number, oy: number }} Transform */

/** @typedef {{ constellations?: boolean, starNames?: boolean, planets?: boolean, detections?: boolean, grid?: boolean }} OverlayLayers */

/**
 * Draw solve overlay geometry in image coordinates (caller sets canvas transform).
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {Transform} transform
 * @param {any | null | undefined} overlay
 * @param {any[] | null | undefined} detections
 * @param {OverlayLayers} layers
 */
export function drawSolveOverlay(ctx, transform, overlay, detections, layers) {
  if (!overlay && (!detections?.length || !layers.detections)) {
    return;
  }

  const inv = 1 / transform.scale;

  if (layers.grid && overlay?.grid?.length) {
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.lineWidth = 1 * inv;
    ctx.lineJoin = 'round';
    for (const segment of overlay.grid) {
      const points = segment.points ?? [];
      for (let i = 1; i < points.length; i += 1) {
        ctx.beginPath();
        ctx.moveTo(points[i - 1][0], points[i - 1][1]);
        ctx.lineTo(points[i][0], points[i][1]);
        ctx.stroke();
      }
    }
  }

  if (layers.constellations && overlay?.constellations?.length) {
    ctx.strokeStyle = 'rgba(94,197,255,0.85)';
    ctx.lineWidth = 1.5 * inv;
    ctx.lineJoin = 'round';
    for (const constellation of overlay.constellations) {
      for (const line of constellation.lines ?? []) {
        for (let i = 1; i < line.length; i += 1) {
          ctx.beginPath();
          ctx.moveTo(line[i - 1][0], line[i - 1][1]);
          ctx.lineTo(line[i][0], line[i][1]);
          ctx.stroke();
        }
      }
    }

    ctx.fillStyle = '#8FD8FF';
    ctx.font = `${13 * inv}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const constellation of overlay.constellations) {
      const label = constellation.label_xy;
      if (!label) {
        continue;
      }
      ctx.fillText(constellation.name ?? constellation.abbr ?? '', label[0], label[1]);
    }
  }

  if (layers.planets && overlay?.planets?.length) {
    const r = 7 * inv;
    ctx.fillStyle = '#FFC24D';
    for (const planet of overlay.planets) {
      if (planet.approx) {
        drawDashedCircle(ctx, planet.x, planet.y, r, inv);
      } else {
        drawDiamond(ctx, planet.x, planet.y, r);
      }
      if (planet.name) {
        const label = planet.approx ? `≈ ${planet.name}` : planet.name;
        drawHaloLabel(ctx, planet.x + r + 4 * inv, planet.y, label, 12 * inv, '#FFC24D');
      }
    }
  }

  if (overlay?.stars?.length) {
    const r = 5 * inv;
    const stroke = 'rgba(255,220,150,0.95)';
    ctx.lineWidth = 1.5 * inv;
    for (const star of overlay.stars) {
      ctx.beginPath();
      ctx.arc(star.x, star.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = stroke;
      ctx.stroke();

      if (layers.starNames && star.label) {
        drawHaloLabel(ctx, star.x + r + 3 * inv, star.y, star.label, 12 * inv, '#FFE9B8');
      }
    }
  }

  if (layers.detections && detections?.length) {
    const r = 6 * inv;
    for (const det of detections) {
      ctx.beginPath();
      ctx.arc(det.x, det.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = det.inlier ? 'rgba(120,255,160,0.9)' : 'rgba(160,170,190,0.6)';
      ctx.lineWidth = 1.5 * inv;
      ctx.stroke();
    }
  }
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {number} r
 */
function drawDiamond(ctx, x, y, r) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
  ctx.fill();
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {number} r
 * @param {number} inv
 */
function drawDashedCircle(ctx, x, y, r, inv) {
  ctx.save();
  ctx.strokeStyle = '#FFC24D';
  ctx.lineWidth = 1.5 * inv;
  ctx.setLineDash([4 * inv, 3 * inv]);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {string} text
 * @param {number} fontSize
 * @param {string} fill
 */
function drawHaloLabel(ctx, x, y, text, fontSize, fill) {
  ctx.font = `${fontSize}px system-ui, sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = Math.max(2, fontSize * 0.18);
  ctx.strokeStyle = 'rgba(0,0,0,0.7)';
  ctx.strokeText(text, x, y);
  ctx.fillStyle = fill;
  ctx.fillText(text, x, y);
}

/**
 * Canvas image viewer with zoom, pan, and fit controls.
 *
 * Later slices may draw overlays in the same transform via `onAfterDraw`.
 */
export class ImageViewer {
  /** @type {HTMLCanvasElement} */
  #canvas;

  /** @type {CanvasRenderingContext2D} */
  #ctx;

  /** @type {HTMLImageElement | null} */
  #img = null;

  /** @type {{ scale: number, ox: number, oy: number }} */
  #state = { scale: 1, ox: 0, oy: 0 };

  /** @type {number} */
  #dpr = 1;

  /** @type {boolean} */
  #dragging = false;

  /** @type {number} */
  #dragStartX = 0;

  /** @type {number} */
  #dragStartY = 0;

  /** @type {number} */
  #dragOx = 0;

  /** @type {number} */
  #dragOy = 0;

  /** @type {((ctx: CanvasRenderingContext2D, transform: Transform) => void) | null} */
  onAfterDraw = null;

  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.#canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('2D canvas context unavailable');
    }
    this.#ctx = ctx;
  }

  /**
   * @param {Element} container
   */
  attach(container) {
    const resize = () => this.#resize(container);
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    this.#canvas.addEventListener('wheel', (event) => this.#onWheel(event), { passive: false });
    this.#canvas.addEventListener('pointerdown', (event) => this.#onPointerDown(event));
    this.#canvas.addEventListener('pointermove', (event) => this.#onPointerMove(event));
    this.#canvas.addEventListener('pointerup', (event) => this.#onPointerUp(event));
    this.#canvas.addEventListener('pointercancel', (event) => this.#onPointerUp(event));
    this.#canvas.addEventListener('dblclick', () => this.fit());
  }

  /**
   * @param {HTMLImageElement} bitmap
   */
  setImage(bitmap) {
    this.#img = bitmap;
    this.fit();
  }

  clearImage() {
    this.#img = null;
    this.redraw();
  }

  fit() {
    if (!this.#img) {
      return;
    }

    const margin = 24;
    const cssW = this.#canvas.clientWidth;
    const cssH = this.#canvas.clientHeight;
    const scaleX = (cssW - margin * 2) / this.#img.naturalWidth;
    const scaleY = (cssH - margin * 2) / this.#img.naturalHeight;
    const scale = Math.min(scaleX, scaleY, 40);

    this.#state.scale = Math.max(scale, 0.05);
    this.#state.ox = (cssW - this.#img.naturalWidth * this.#state.scale) / 2;
    this.#state.oy = (cssH - this.#img.naturalHeight * this.#state.scale) / 2;
    this.redraw();
  }

  oneToOne() {
    if (!this.#img) {
      return;
    }

    const cssW = this.#canvas.clientWidth;
    const cssH = this.#canvas.clientHeight;
    this.#state.scale = 1;
    this.#state.ox = (cssW - this.#img.naturalWidth) / 2;
    this.#state.oy = (cssH - this.#img.naturalHeight) / 2;
    this.redraw();
  }

  /**
   * @param {number} factor
   * @param {number | null} [anchorX]
   * @param {number | null} [anchorY]
   */
  zoomBy(factor, anchorX = null, anchorY = null) {
    if (!this.#img) {
      return;
    }

    const rect = this.#canvas.getBoundingClientRect();
    const px = anchorX ?? rect.width / 2;
    const py = anchorY ?? rect.height / 2;

    const worldX = (px - this.#state.ox) / this.#state.scale;
    const worldY = (py - this.#state.oy) / this.#state.scale;

    const nextScale = Math.min(40, Math.max(0.05, this.#state.scale * factor));
    this.#state.scale = nextScale;
    this.#state.ox = px - worldX * nextScale;
    this.#state.oy = py - worldY * nextScale;
    this.redraw();
  }

  redraw() {
    const cssW = this.#canvas.clientWidth;
    const cssH = this.#canvas.clientHeight;
    this.#ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.#ctx.clearRect(0, 0, this.#canvas.width, this.#canvas.height);

    if (!this.#img) {
      return;
    }

    this.#ctx.imageSmoothingEnabled = this.#state.scale <= 3;
    this.#ctx.setTransform(
      this.#state.scale * this.#dpr,
      0,
      0,
      this.#state.scale * this.#dpr,
      this.#state.ox * this.#dpr,
      this.#state.oy * this.#dpr,
    );
    this.#ctx.drawImage(this.#img, 0, 0);

    if (this.onAfterDraw) {
      this.#ctx.save();
      this.#ctx.beginPath();
      this.#ctx.rect(0, 0, this.#img.width, this.#img.height);
      this.#ctx.clip();
      this.onAfterDraw(this.#ctx, { ...this.#state });
      this.#ctx.restore();
    }
  }

  /**
   * @param {Element} container
   */
  #resize(container) {
    this.#dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(container.clientWidth));
    const height = Math.max(1, Math.floor(container.clientHeight));
    this.#canvas.width = Math.floor(width * this.#dpr);
    this.#canvas.height = Math.floor(height * this.#dpr);
    this.#canvas.style.width = `${width}px`;
    this.#canvas.style.height = `${height}px`;
    this.redraw();
  }

  /**
   * @param {WheelEvent} event
   */
  #onWheel(event) {
    if (!this.#img) {
      return;
    }
    event.preventDefault();
    const rect = this.#canvas.getBoundingClientRect();
    const factor = Math.exp(-event.deltaY * 0.0015);
    this.zoomBy(factor, event.clientX - rect.left, event.clientY - rect.top);
  }

  /**
   * @param {PointerEvent} event
   */
  #onPointerDown(event) {
    if (!this.#img || event.button !== 0) {
      return;
    }
    this.#dragging = true;
    this.#dragStartX = event.clientX;
    this.#dragStartY = event.clientY;
    this.#dragOx = this.#state.ox;
    this.#dragOy = this.#state.oy;
    this.#canvas.setPointerCapture(event.pointerId);
    this.#canvas.closest('.canvas-area')?.classList.add('is-dragging');
  }

  /**
   * @param {PointerEvent} event
   */
  #onPointerMove(event) {
    if (!this.#dragging) {
      return;
    }
    const dx = event.clientX - this.#dragStartX;
    const dy = event.clientY - this.#dragStartY;
    this.#state.ox = this.#dragOx + dx;
    this.#state.oy = this.#dragOy + dy;
    this.redraw();
  }

  /**
   * @param {PointerEvent} event
   */
  #onPointerUp(event) {
    if (!this.#dragging) {
      return;
    }
    this.#dragging = false;
    if (this.#canvas.hasPointerCapture(event.pointerId)) {
      this.#canvas.releasePointerCapture(event.pointerId);
    }
    this.#canvas.closest('.canvas-area')?.classList.remove('is-dragging');
  }
}
