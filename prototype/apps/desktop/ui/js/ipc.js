/** @returns {boolean} */
export function hasTauri() {
  return typeof window.__TAURI__ !== 'undefined';
}

/**
 * @param {string} command
 * @param {Record<string, unknown>} [args]
 */
export function invoke(command, args = {}) {
  if (!hasTauri()) {
    throw new Error('Tauri API unavailable');
  }
  return window.__TAURI__.core.invoke(command, args);
}

/** @type {Map<string, string>} */
const imageUrlCache = new Map();

/**
 * @param {number} id
 * @param {'raw' | 'stretched'} view
 * @returns {Promise<string>}
 */
export async function imageUrl(id, view) {
  const key = `${id}:${view}`;
  const existing = imageUrlCache.get(key);
  if (existing) {
    URL.revokeObjectURL(existing);
  }

  const bytes = await invoke('image_png', { id, view });
  const blob = new Blob([bytes], { type: 'image/png' });
  const url = URL.createObjectURL(blob);
  imageUrlCache.set(key, url);
  return url;
}

/** @param {number} id */
export function revokeImageUrls(id) {
  for (const view of ['raw', 'stretched']) {
    const key = `${id}:${view}`;
    const cached = imageUrlCache.get(key);
    if (cached) {
      URL.revokeObjectURL(cached);
      imageUrlCache.delete(key);
    }
  }
}

/**
 * @param {number} id
 * @param {number} utcOffsetHours
 * @param {(event: { stage: string, detail?: string | null }) => void} onProgress
 */
export async function solveImage(id, utcOffsetHours, onProgress) {
  const channel = new window.__TAURI__.core.Channel();
  channel.onmessage = onProgress;
  return invoke('solve_image', { id, utcOffsetHours, onProgress: channel });
}
