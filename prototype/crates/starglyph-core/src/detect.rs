//! Star detection for grayscale frame images.

use rayon::prelude::*;

use crate::image_input::FrameImage;

const MAD_TO_SIGMA: f32 = 1.4826;

/// Noise floor for MAD-based sigma estimates, in normalized gray units.
///
/// Heavily denoised consumer JPEGs (phone Night mode, processed DSLR exports)
/// clip the sky to exact black: ~90% of pixels equal 0, the MAD collapses and
/// the detection threshold drops to ~0, turning residual JPEG noise into
/// thousands of merged candidate blobs. All inputs are normalized through
/// 8-bit luma, so one quantization step (1/255) is the smallest scale at which
/// signal is distinguishable at all; sigma is never estimated below it.
const RAW_SIGMA_FLOOR: f32 = 1.0 / 255.0;

/// Blobs at least this large are screened by the peak-concentration test;
/// smaller point sources have too few pixels for the statistic to be stable.
const CONCENTRATION_MIN_AREA: u32 = 25;

/// Maximum fraction of pixels the threshold mask may select.
///
/// Gaussian noise puts ~0.6% of pixels above 2.5σ; real starfields add well
/// under 1%. When the mask covers more, the threshold is not separating
/// sources from background — on denoised JPEGs the faint PSF skirts of dense
/// starfields (or Milky-Way texture) connect into big merged networks and
/// per-blob reasoning breaks down. The threshold is then doubled (up to
/// [`MAX_THRESHOLD_DOUBLINGS`]) until the mask is sparse: star cores are far
/// brighter than the junk floor, so they survive every doubling.
const MAX_MASK_FILL: f32 = 0.02;
/// Upper bound on adaptive threshold doublings (16× total).
const MAX_THRESHOLD_DOUBLINGS: u32 = 4;

/// Configuration for the star-detection pipeline.
#[derive(Debug, Clone)]
pub struct DetectConfig {
    /// Mesh cell size in pixels for background estimation.
    pub mesh_px: u32,
    /// Gaussian kernel sigma in pixels for matched filtering.
    pub sigma_px: f32,
    /// Detection threshold in units of convolved noise sigma.
    pub k_sigma: f32,
    /// Border margin in pixels; blobs touching it are rejected.
    pub border_px: u32,
    /// Minimum core area (pixels above 3σ_raw) to accept a blob.
    pub min_core_area: u32,
    /// Peak residual threshold in units of σ_raw.
    pub k_sigma_peak: f32,
    /// Maximum blob area; larger blobs are rejected as extended objects.
    pub max_area: u32,
    /// Maximum elongation (sqrt eigenvalue ratio) for area ≥ 6.
    pub max_elongation: f32,
    /// Maximum number of detections to return.
    pub max_detections: u32,
    /// Peak residual (normalized units) above which a blob counts as a
    /// saturated source rather than diffuse structure.
    pub sat_min_peak: f32,
    /// Area cap for saturated round blobs. Bright-star cores with bloom/halo
    /// on consumer JPEGs far exceed `max_area` (a mag −1 star can span
    /// hundreds of pixels at 12 MP); round saturated blobs are stars, and
    /// dropping them removes exactly the anchors pattern matching needs.
    pub sat_max_area: u32,
    /// Minimum peak-to-mean concentration (`peak · area / flux`) for blobs of
    /// area ≥ [`CONCENTRATION_MIN_AREA`] that are not saturated. Point sources
    /// are strongly peaked; Milky-Way star clouds and nebulosity are flat and
    /// otherwise dominate the flux ranking on dense-field frames.
    pub min_concentration: f32,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            mesh_px: 32,
            sigma_px: 1.2,
            k_sigma: 2.5,
            border_px: 8,
            min_core_area: 2,
            k_sigma_peak: 4.0,
            max_area: 150,
            max_elongation: 2.5,
            max_detections: 40,
            sat_min_peak: 0.7,
            sat_max_area: 1500,
            min_concentration: 2.5,
        }
    }
}

/// A single star detection with sub-pixel centroid and photometry.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Sub-pixel x coordinate (column).
    pub x: f64,
    /// Sub-pixel y coordinate (row).
    pub y: f64,
    /// Total flux (sum of positive residuals over blob + ring).
    pub flux: f32,
    /// Peak raw residual in the blob.
    pub peak: f32,
    /// Signal-to-noise ratio.
    pub snr: f32,
    /// Blob area in pixels.
    pub area: u32,
    /// Elongation (sqrt of eigenvalue ratio).
    pub elongation: f32,
    /// Rank by flux (0 = brightest).
    pub rank: u32,
}

/// Per-run detection statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectStats {
    /// Raw noise sigma (MAD-based).
    pub sigma: f32,
    /// Median of the input frame gray values.
    pub background_median: f32,
    /// Total connected-component candidates before filtering.
    pub candidates: u32,
    pub rejected_border: u32,
    pub rejected_small: u32,
    pub rejected_large: u32,
    pub rejected_elongated: u32,
    pub rejected_faint: u32,
    /// Large low-concentration blobs (star clouds, nebulosity).
    pub rejected_diffuse: u32,
    pub accepted: u32,
}

/// Result of star detection.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectResult {
    pub detections: Vec<Detection>,
    pub stats: DetectStats,
}

/// Run the full star-detection pipeline on a frame.
pub fn detect_stars(frame: &FrameImage, config: &DetectConfig) -> DetectResult {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let n = width * height;

    let background_median = median_f32(&frame.gray);

    // Step 1: column/row artifact removal
    let mut resid = remove_column_row_artifacts(&frame.gray, width, height);

    // Step 2: mesh background subtraction
    let bg = mesh_background(&resid, width, height, config.mesh_px);
    for (r, b) in resid.iter_mut().zip(bg.iter()) {
        *r -= b;
    }

    // Step 3: noise estimate. When the MAD collapses (clipped-black denoised
    // JPEGs: most pixels exactly 0) fall back to an upper-quantile spread that
    // still sees the residual junk level (JPEG artifacts, haze plateaus), and
    // never go below the 8-bit quantization step; see robust_noise_sigma.
    let sigma_raw = robust_noise_sigma(&resid, width, height, config.border_px, RAW_SIGMA_FLOOR);

    // Step 4: matched filter + threshold + connected components
    let kernel = gaussian_kernel_7x7(config.sigma_px);
    let conv = convolve_7x7(&resid, width, height, &kernel);
    // Convolving i.i.d. noise scales sigma by the kernel L2 norm; floor the
    // convolved estimate consistently with the raw floor.
    let kernel_l2 = kernel.iter().flatten().map(|v| v * v).sum::<f32>().sqrt();
    let sigma_conv = robust_noise_sigma(
        &conv,
        width,
        height,
        config.border_px,
        RAW_SIGMA_FLOOR * kernel_l2,
    );
    let mut threshold = config.k_sigma * sigma_conv;

    // Adaptive threshold: keep the mask sparse (see MAX_MASK_FILL).
    let mut mask: Vec<bool> = conv.iter().map(|&v| v > threshold).collect();
    for _ in 0..MAX_THRESHOLD_DOUBLINGS {
        let fill = mask.iter().filter(|&&m| m).count() as f32 / n.max(1) as f32;
        if fill <= MAX_MASK_FILL {
            break;
        }
        threshold *= 2.0;
        for (m, &v) in mask.iter_mut().zip(conv.iter()) {
            *m = v > threshold;
        }
    }
    let blobs = label_components(&mask, width, height);

    let mut stats = DetectStats {
        sigma: sigma_raw,
        background_median,
        candidates: blobs.len() as u32,
        rejected_border: 0,
        rejected_small: 0,
        rejected_large: 0,
        rejected_elongated: 0,
        rejected_faint: 0,
        rejected_diffuse: 0,
        accepted: 0,
    };

    let border = config.border_px as i32;
    let w = frame.width as i32;
    let h = frame.height as i32;
    let core_thresh = 3.0 * sigma_raw;
    let peak_thresh = config.k_sigma_peak * sigma_raw;

    let mut detections = Vec::new();

    for blob in blobs {
        let area = blob.pixels.len() as u32;

        // Border rejection
        if blob.min_x < border
            || blob.min_y < border
            || blob.max_x >= w - border
            || blob.max_y >= h - border
        {
            stats.rejected_border += 1;
            continue;
        }

        // Core area
        let core_area = blob
            .pixels
            .iter()
            .filter(|&&(x, y)| resid[y as usize * width + x as usize] > core_thresh)
            .count() as u32;
        if core_area < config.min_core_area {
            stats.rejected_small += 1;
            continue;
        }

        // Peak
        let peak = blob
            .pixels
            .iter()
            .map(|&(x, y)| resid[y as usize * width + x as usize])
            .fold(f32::NEG_INFINITY, f32::max);
        if peak < peak_thresh {
            stats.rejected_faint += 1;
            continue;
        }

        let saturated = peak >= config.sat_min_peak;

        // Oversized blob: a bright star on a consumer JPEG (bloom, diffusion
        // halo) can far exceed max_area, but its *half-peak core* stays
        // compact — while flat extended structure (clouds, nebulosity) keeps
        // most of its footprint at half peak. Re-segment and judge the core.
        let measured: Blob = if area > config.max_area {
            let half_peak = 0.5 * peak;
            let core_pixels: Vec<(i32, i32)> = blob
                .pixels
                .iter()
                .copied()
                .filter(|&(x, y)| resid[y as usize * width + x as usize] >= half_peak)
                .collect();
            let core_cap = if saturated {
                config.sat_max_area
            } else {
                config.max_area
            };
            if core_pixels.is_empty() || core_pixels.len() as u32 > core_cap {
                stats.rejected_large += 1;
                continue;
            }
            blob_from_pixels(core_pixels)
        } else {
            blob
        };

        let elongation = blob_elongation(&measured, &resid, width);
        if measured.pixels.len() >= 6 && elongation > config.max_elongation {
            stats.rejected_elongated += 1;
            continue;
        }

        let (cx, cy, flux) = refine_centroid(&measured, &resid, width, height);
        let snr = flux / (sigma_raw * (area as f32).sqrt());

        // Diffuse-structure screen for mid-size blobs that never went through
        // core re-segmentation: a point source concentrates its light (high
        // peak relative to mean surface brightness); star-cloud and nebulosity
        // fragments are flat. Saturated blobs are exempt — a clipped flat core
        // legitimately has concentration ≈ 1, and so does a re-segmented
        // half-peak core (its compactness at half peak is already the test).
        if area <= config.max_area && area >= CONCENTRATION_MIN_AREA && !saturated {
            let concentration = peak * area as f32 / flux.max(1e-9);
            if concentration < config.min_concentration {
                stats.rejected_diffuse += 1;
                continue;
            }
        }

        detections.push(Detection {
            x: cx,
            y: cy,
            flux,
            peak,
            snr,
            area,
            elongation,
            rank: 0,
        });
        stats.accepted += 1;
    }

    // Step 7: deduplicate, sort, truncate, rank
    deduplicate_detections(&mut detections, 3.0);
    detections.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    detections.truncate(config.max_detections as usize);
    for (rank, det) in detections.iter_mut().enumerate() {
        det.rank = rank as u32;
    }

    let _ = n; // silence unused warning in edge cases
    DetectResult { detections, stats }
}

// --- Pipeline helpers ---

fn remove_column_row_artifacts(gray: &[f32], width: usize, height: usize) -> Vec<f32> {
    // Column medians first, then row medians of the column-subtracted data —
    // the same math as the sequential version; columns and rows are
    // independent, so each level parallelizes bit-exactly.
    let col_medians: Vec<f32> = (0..width)
        .into_par_iter()
        .map(|x| {
            let col: Vec<f32> = (0..height).map(|y| gray[y * width + x]).collect();
            median_f32(&col)
        })
        .collect();

    let mut out = vec![0.0f32; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let src = &gray[y * width..(y + 1) * width];
        for (x, (dst, &g)) in row.iter_mut().zip(src.iter()).enumerate() {
            *dst = g - col_medians[x];
        }
        let med = median_f32(row);
        for v in row.iter_mut() {
            *v -= med;
        }
    });
    out
}

fn mesh_background(data: &[f32], width: usize, height: usize, mesh_px: u32) -> Vec<f32> {
    let mesh = mesh_px as usize;
    let n_cols = width.div_ceil(mesh);
    let n_rows = height.div_ceil(mesh);

    // Mesh cells and interpolated rows are independent — both levels run in
    // parallel with per-cell/per-pixel arithmetic unchanged.
    let mut grid = vec![0.0f32; n_cols * n_rows];
    grid.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row, grid_row)| {
            for (col, cell_median) in grid_row.iter_mut().enumerate() {
                let x0 = col * mesh;
                let y0 = row * mesh;
                let x1 = ((col + 1) * mesh).min(width);
                let y1 = ((row + 1) * mesh).min(height);

                let mut cell = Vec::with_capacity((x1 - x0) * (y1 - y0));
                for y in y0..y1 {
                    for x in x0..x1 {
                        cell.push(data[y * width + x]);
                    }
                }
                *cell_median = sigma_clipped_median(&cell, 3, 3.0);
            }
        });

    // Bilinear interpolation from cell centers
    let mut bg = vec![0.0f32; width * height];
    bg.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, bg_row)| {
            let fy = (y as f64 + 0.5) / mesh as f64 - 0.5;
            for (x, v) in bg_row.iter_mut().enumerate() {
                let fx = (x as f64 + 0.5) / mesh as f64 - 0.5;
                *v = bilinear_grid(&grid, n_cols, n_rows, fx, fy);
            }
        });
    bg
}

fn bilinear_grid(grid: &[f32], n_cols: usize, n_rows: usize, fx: f64, fy: f64) -> f32 {
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let tx = (fx - x0 as f64) as f32;
    let ty = (fy - y0 as f64) as f32;

    let sample = |cx: i32, cy: i32| -> f32 {
        let cx = cx.clamp(0, n_cols as i32 - 1) as usize;
        let cy = cy.clamp(0, n_rows as i32 - 1) as usize;
        grid[cy * n_cols + cx]
    };

    let v00 = sample(x0, y0);
    let v10 = sample(x1, y0);
    let v01 = sample(x0, y1);
    let v11 = sample(x1, y1);

    let top = v00 * (1.0 - tx) + v10 * tx;
    let bot = v01 * (1.0 - tx) + v11 * tx;
    top * (1.0 - ty) + bot * ty
}

fn noise_sigma(data: &[f32], width: usize, height: usize, border_px: u32) -> f32 {
    let b = border_px as usize;
    let mut samples = Vec::new();
    for y in b..height.saturating_sub(b) {
        for x in b..width.saturating_sub(b) {
            samples.push(data[y * width + x]);
        }
    }
    if samples.is_empty() {
        return mad_to_sigma(mad_f32(data));
    }
    mad_to_sigma(mad_f32(&samples))
}

/// MAD-based sigma with two fallbacks for clipped data, never below `floor`.
///
/// On denoised consumer JPEGs the sky is clipped to exact black: with > 50% of
/// pixels identical the MAD is 0 and says nothing. The upper-quantile spread
/// `(Q(97.72%) − median) / 2` — equal to sigma for Gaussian noise — still sees
/// the residual junk level (JPEG artifacts, haze plateaus) as long as < 97.7%
/// of pixels are clipped. It is only consulted when the MAD collapses below
/// `floor`, so frames with healthy noise keep today's MAD behavior.
fn robust_noise_sigma(
    data: &[f32],
    width: usize,
    height: usize,
    border_px: u32,
    floor: f32,
) -> f32 {
    let mad_sigma = noise_sigma(data, width, height, border_px);
    if mad_sigma >= floor {
        return mad_sigma;
    }
    let b = border_px as usize;
    let mut samples: Vec<f32> =
        Vec::with_capacity(height.saturating_sub(2 * b) * width.saturating_sub(2 * b));
    for y in b..height.saturating_sub(b) {
        for x in b..width.saturating_sub(b) {
            samples.push(data[y * width + x]);
        }
    }
    if samples.is_empty() {
        return floor;
    }
    // Selection instead of a full sort: the same order statistics in O(n).
    let last = samples.len() - 1;
    let i_median = (last as f64 * 0.5) as usize;
    let i_upper = (last as f64 * 0.9772) as usize;
    let median = *samples.select_nth_unstable_by(i_median, cmp_f32).1;
    let upper = *samples.select_nth_unstable_by(i_upper, cmp_f32).1;
    let quantile_sigma = (upper - median) / 2.0;
    quantile_sigma.max(floor)
}

fn gaussian_kernel_7x7(sigma: f32) -> [[f32; 7]; 7] {
    let mut kernel = [[0.0f32; 7]; 7];
    let mut sum = 0.0f32;
    for dy in -3i32..=3 {
        for dx in -3i32..=3 {
            let r2 = (dx * dx + dy * dy) as f32;
            let v = (-r2 / (2.0 * sigma * sigma)).exp();
            kernel[(dy + 3) as usize][(dx + 3) as usize] = v;
            sum += v;
        }
    }
    for row in &mut kernel {
        for v in row {
            *v /= sum;
        }
    }
    kernel
}

fn convolve_7x7(data: &[f32], width: usize, height: usize, kernel: &[[f32; 7]; 7]) -> Vec<f32> {
    // The dominant O(49·n) pass of the pipeline. Output rows are independent
    // and per-pixel arithmetic is unchanged → parallel result is bit-exact.
    let mut out = vec![0.0f32; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, out_px) in row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for dy in -3i32..=3 {
                for dx in -3i32..=3 {
                    let sx = x as i32 + dx;
                    let sy = y as i32 + dy;
                    if sx >= 0 && sy >= 0 && sx < width as i32 && sy < height as i32 {
                        let kv = kernel[(dy + 3) as usize][(dx + 3) as usize];
                        sum += data[sy as usize * width + sx as usize] * kv;
                    }
                }
            }
            *out_px = sum;
        }
    });
    out
}

struct Blob {
    pixels: Vec<(i32, i32)>,
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
}

fn label_components(mask: &[bool], width: usize, height: usize) -> Vec<Blob> {
    let n = width * height;
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0u8; n];

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            let root = find(parent, parent[i]);
            parent[i] = root;
        }
        parent[i]
    }

    fn unite(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        if rank[ra] < rank[rb] {
            parent[ra] = rb;
        } else if rank[ra] > rank[rb] {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra] += 1;
        }
    }

    let idx = |x: usize, y: usize| y * width + x;

    for y in 0..height {
        for x in 0..width {
            if !mask[idx(x, y)] {
                continue;
            }
            let i = idx(x, y);
            for &(dx, dy) in &[(1i32, 0), (0, 1), (1, 1), (1, -1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                    let ni = idx(nx as usize, ny as usize);
                    if mask[ni] {
                        unite(&mut parent, &mut rank, i, ni);
                    }
                }
            }
        }
    }

    use std::collections::HashMap;
    let mut groups: HashMap<usize, Vec<(i32, i32)>> = HashMap::new();
    for y in 0..height {
        for x in 0..width {
            if !mask[idx(x, y)] {
                continue;
            }
            let root = find(&mut parent, idx(x, y));
            groups.entry(root).or_default().push((x as i32, y as i32));
        }
    }

    groups.into_values().map(blob_from_pixels).collect()
}

fn blob_from_pixels(pixels: Vec<(i32, i32)>) -> Blob {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for &(x, y) in &pixels {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Blob {
        pixels,
        min_x,
        min_y,
        max_x,
        max_y,
    }
}

fn blob_elongation(blob: &Blob, resid: &[f32], width: usize) -> f32 {
    let mut mxx = 0.0f64;
    let mut myy = 0.0f64;
    let mut mxy = 0.0f64;
    let mut wsum = 0.0f64;
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;

    for &(x, y) in &blob.pixels {
        let w = resid[y as usize * width + x as usize].max(0.0) as f64;
        if w > 0.0 {
            cx += w * x as f64;
            cy += w * y as f64;
            wsum += w;
        }
    }
    if wsum <= 0.0 {
        return 1.0;
    }
    cx /= wsum;
    cy /= wsum;

    for &(x, y) in &blob.pixels {
        let w = resid[y as usize * width + x as usize].max(0.0) as f64;
        if w > 0.0 {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            mxx += w * dx * dx;
            myy += w * dy * dy;
            mxy += w * dx * dy;
        }
    }
    mxx /= wsum;
    myy /= wsum;
    mxy /= wsum;

    let trace = mxx + myy;
    let det = mxx * myy - mxy * mxy;
    let disc = (trace * trace / 4.0 - det).max(0.0);
    let lambda1 = trace / 2.0 + disc.sqrt();
    let lambda2 = (trace / 2.0 - disc.sqrt()).max(1e-12);
    (lambda1 / lambda2).sqrt() as f32
}

fn refine_centroid(blob: &Blob, resid: &[f32], width: usize, height: usize) -> (f64, f64, f32) {
    let pixel_set: std::collections::HashSet<(i32, i32)> = blob.pixels.iter().copied().collect();

    // Collect blob + 1px dilation ring with positive resid
    let mut samples: Vec<(i32, i32, f32)> = Vec::new();
    for &(x, y) in &blob.pixels {
        let v = resid[y as usize * width + x as usize];
        if v > 0.0 {
            samples.push((x, y, v));
        }
    }
    for &(x, y) in &blob.pixels {
        for &(dx, dy) in &[
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ] {
            let nx = x + dx;
            let ny = y + dy;
            if pixel_set.contains(&(nx, ny)) {
                continue;
            }
            if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                let v = resid[ny as usize * width + nx as usize];
                if v > 0.0 {
                    samples.push((nx, ny, v));
                }
            }
        }
    }

    let mut wsum = 0.0f64;
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    let mut flux = 0.0f32;
    for &(x, y, w) in &samples {
        let wd = w as f64;
        cx += wd * x as f64;
        cy += wd * y as f64;
        wsum += wd;
        flux += w;
    }
    if wsum > 0.0 {
        cx /= wsum;
        cy /= wsum;
    }

    // Windowed centroid refinement: 3 iterations, 11×11, σ_w = 1.5
    let sigma_w = 1.5f64;
    let half = 5i32;
    for _ in 0..3 {
        let mut nwsum = 0.0f64;
        let mut ncx = 0.0f64;
        let mut ncy = 0.0f64;
        let icx = cx.round() as i32;
        let icy = cy.round() as i32;
        for dy in -half..=half {
            for dx in -half..=half {
                let px = icx + dx;
                let py = icy + dy;
                if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                    continue;
                }
                let r = resid[py as usize * width + px as usize];
                if r <= 0.0 {
                    continue;
                }
                let dist2 = (px as f64 - cx).powi(2) + (py as f64 - cy).powi(2);
                let gw = (-dist2 / (2.0 * sigma_w * sigma_w)).exp();
                let w = r as f64 * gw;
                ncx += w * px as f64;
                ncy += w * py as f64;
                nwsum += w;
            }
        }
        if nwsum > 0.0 {
            cx = ncx / nwsum;
            cy = ncy / nwsum;
        }
    }

    (cx, cy, flux)
}

fn deduplicate_detections(detections: &mut Vec<Detection>, min_dist: f64) {
    detections.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Detection> = Vec::new();
    'outer: for det in detections.drain(..) {
        for k in &kept {
            let dx = det.x - k.x;
            let dy = det.y - k.y;
            if (dx * dx + dy * dy).sqrt() < min_dist {
                continue 'outer;
            }
        }
        kept.push(det);
    }
    *detections = kept;
}

// --- Statistics helpers ---

fn cmp_f32(a: &f32, b: &f32) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    // Selection instead of a full sort: this runs over whole frames and every
    // column/row/mesh-cell, and picks exactly the values a sort would.
    let mut work = values.to_vec();
    let mid = work.len() / 2;
    let (below, upper, _) = work.select_nth_unstable_by(mid, cmp_f32);
    let upper = *upper;
    if values.len().is_multiple_of(2) {
        // `sorted[mid - 1]` is the maximum of the left partition.
        let lower = below.iter().copied().max_by(cmp_f32).unwrap_or(upper);
        (lower + upper) / 2.0
    } else {
        upper
    }
}

fn mad_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let med = median_f32(values);
    let devs: Vec<f32> = values.iter().map(|v| (v - med).abs()).collect();
    median_f32(&devs)
}

fn mad_to_sigma(mad: f32) -> f32 {
    mad * MAD_TO_SIGMA
}

fn sigma_clipped_median(values: &[f32], iterations: usize, clip_sigma: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut work = values.to_vec();
    for _ in 0..iterations {
        let result = median_f32(&work);
        let sigma = mad_to_sigma(mad_f32(&work));
        let lo = result - clip_sigma * sigma;
        let hi = result + clip_sigma * sigma;
        work.retain(|&v| v >= lo && v <= hi);
        if work.is_empty() {
            return result;
        }
    }
    median_f32(&work)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn make_frame(width: u32, height: u32, gray: Vec<f32>) -> FrameImage {
        FrameImage {
            width,
            height,
            gray,
            source_name: "test".to_string(),
            exif: None,
        }
    }

    /// Flat background with deterministic ripple so MAD-based noise estimates are non-zero.
    fn flat_background(width: u32, height: u32, level: f32) -> Vec<f32> {
        let mut gray = vec![0.0f32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let ripple = 0.012 * ((x as f32 * 0.73).sin() * (y as f32 * 0.51).sin());
                gray[(y * width + x) as usize] = level + ripple;
            }
        }
        gray
    }

    fn gaussian_bump(
        width: u32,
        height: u32,
        cx: f64,
        cy: f64,
        amp: f32,
        sigma: f32,
        base: &mut [f32],
    ) {
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let r2 = (dx * dx + dy * dy) as f32;
                base[(y * width + x) as usize] += amp * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    /// Deterministic xorshift noise (tests must not depend on an RNG crate).
    struct XorShift(u32);
    impl XorShift {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// Star field with known truth: windowed Gaussian PSFs (beyond ±8 px a
    /// σ≈1.3 kernel is ~e⁻¹⁹) over a noisy pedestal; positions follow a
    /// low-discrepancy sequence so spacing stays star-like without an RNG.
    fn synthetic_star_field(
        width: u32,
        height: u32,
        n_stars: usize,
        noise_sigma: f32,
        seed: u32,
    ) -> (Vec<f32>, Vec<(f64, f64)>) {
        let mut rng = XorShift(seed);
        let mut gray = vec![0.08f32; (width as usize) * (height as usize)];
        for v in gray.iter_mut() {
            *v += noise_sigma * rng.next_f32();
        }
        let sigma = 1.3f32;
        let win = 8i64;
        let margin = 16.0;
        let mut truth = Vec::with_capacity(n_stars);
        for i in 0..n_stars {
            let hx = (i as f64 * 0.754_877_666_2).fract();
            let hy = (i as f64 * 0.569_840_290_998).fract();
            let cx = margin + hx * (f64::from(width) - 2.0 * margin);
            let cy = margin + hy * (f64::from(height) - 2.0 * margin);
            let amp = 0.15 + 0.075 * ((i % 10) as f32);
            let (icx, icy) = (cx.round() as i64, cy.round() as i64);
            for y in (icy - win).max(0)..=(icy + win).min(i64::from(height) - 1) {
                for x in (icx - win).max(0)..=(icx + win).min(i64::from(width) - 1) {
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    let r2 = (dx * dx + dy * dy) as f32;
                    gray[y as usize * width as usize + x as usize] +=
                        amp * (-r2 / (2.0 * sigma * sigma)).exp();
                }
            }
            truth.push((cx, cy));
        }
        (gray, truth)
    }

    /// Direct detection-quality harness (B3): precision, recall and centroid
    /// RMS against injected truth — regressions in the detector surface here
    /// before they show up as solve-rate.
    #[test]
    fn detection_precision_recall_on_synthetic_field() {
        let (width, height) = (1024u32, 768u32);
        let (gray, truth) = synthetic_star_field(width, height, 60, 0.02, 0xC0FF_EE01);
        let frame = make_frame(width, height, gray);
        let config = DetectConfig {
            max_detections: 100,
            ..DetectConfig::default()
        };
        let result = detect_stars(&frame, &config);

        let mut truth_hit = vec![false; truth.len()];
        let mut true_positives = 0u32;
        let mut centroid_sq = 0.0f64;
        for det in &result.detections {
            let mut best: Option<(usize, f64)> = None;
            for (ti, &(tx, ty)) in truth.iter().enumerate() {
                if truth_hit[ti] {
                    continue;
                }
                let d2 = (det.x - tx).powi(2) + (det.y - ty).powi(2);
                if d2 <= 4.0 && best.is_none_or(|(_, bd)| d2 < bd) {
                    best = Some((ti, d2));
                }
            }
            if let Some((ti, d2)) = best {
                truth_hit[ti] = true;
                true_positives += 1;
                centroid_sq += d2;
            }
        }
        let precision = f64::from(true_positives) / result.detections.len().max(1) as f64;
        let recall = f64::from(true_positives) / truth.len() as f64;
        let centroid_rms = (centroid_sq / f64::from(true_positives.max(1))).sqrt();
        assert!(
            precision >= 0.95,
            "precision {precision:.3} ({true_positives}/{})",
            result.detections.len()
        );
        assert!(recall >= 0.90, "recall {recall:.3}");
        assert!(centroid_rms <= 0.35, "centroid rms {centroid_rms:.3}px");
    }

    /// Perf budget (B3): full detection on a 12 Mpx frame within 2.0 s.
    /// Meaningless in debug builds, hence ignored:
    /// `cargo test -p starglyph-core --release -- --ignored detect_12mp`
    #[test]
    #[ignore = "perf benchmark; run with --release"]
    fn detect_12mp_within_two_seconds() {
        let (width, height) = (4000u32, 3000u32);
        let (gray, _) = synthetic_star_field(width, height, 300, 0.02, 0xBEEF_0001);
        let frame = make_frame(width, height, gray);
        let started = std::time::Instant::now();
        let result = detect_stars(&frame, &DetectConfig::default());
        let elapsed = started.elapsed();
        eprintln!(
            "12 Mpx detect: {} ms, {} detections",
            elapsed.as_millis(),
            result.detections.len()
        );
        assert!(
            result.detections.len() >= 40,
            "a rich field should saturate the default cap, got {}",
            result.detections.len()
        );
        assert!(
            elapsed.as_secs_f64() <= 2.0,
            "12 Mpx detect took {:.2}s (budget 2.0s)",
            elapsed.as_secs_f64()
        );
    }

    #[test]
    fn single_gaussian_star_detected_near_truth() {
        let width = 128u32;
        let height = 128u32;
        let cx = 64.5f64;
        let cy = 60.3f64;
        let mut gray = flat_background(width, height, 0.15);
        gaussian_bump(width, height, cx, cy, 0.5, 1.0, &mut gray);
        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &DetectConfig::default());

        assert_eq!(result.detections.len(), 1, "expected 1 detection");
        let det = &result.detections[0];
        assert!(
            (det.x - cx).abs() < 0.15,
            "x centroid off: {} vs {}",
            det.x,
            cx
        );
        assert!(
            (det.y - cy).abs() < 0.15,
            "y centroid off: {} vs {}",
            det.y,
            cy
        );
    }

    #[test]
    fn hot_single_pixel_rejected() {
        let width = 64u32;
        let height = 64u32;
        let mut gray = flat_background(width, height, 0.15);
        gray[32 * width as usize + 32] = 0.9;
        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &DetectConfig::default());

        assert!(
            result.detections.is_empty(),
            "hot pixel should be rejected, got {:?}",
            result.detections
        );
        assert!(result.stats.rejected_small >= 1 || result.stats.rejected_faint >= 1);
    }

    #[test]
    fn hot_column_produces_no_detections_along_it() {
        let width = 64u32;
        let height = 64u32;
        // Add a real star away from the column
        let mut gray = flat_background(width, height, 0.05);
        for y in 0..height {
            gray[y as usize * width as usize + 4] = 0.6;
        }
        gaussian_bump(width, height, 40.0, 30.0, 0.4, 1.0, &mut gray);

        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &DetectConfig::default());

        for det in &result.detections {
            assert!(
                (det.x - 4.0).abs() > 3.0,
                "detection at hot column: x={}",
                det.x
            );
        }
    }

    #[test]
    fn star_touching_border_margin_rejected() {
        let width = 64u32;
        let height = 64u32;
        let config = DetectConfig {
            border_px: 8,
            ..DetectConfig::default()
        };
        // Star near top-left corner inside border margin
        let mut gray = flat_background(width, height, 0.15);
        gaussian_bump(width, height, 4.0, 4.0, 0.5, 1.0, &mut gray);
        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &config);

        assert!(result.detections.is_empty());
        assert!(result.stats.rejected_border >= 1);
    }

    #[test]
    fn large_blob_rejected() {
        let width = 128u32;
        let height = 128u32;
        let mut gray = flat_background(width, height, 0.15);
        // ~484 px solid blob (below 500 but above max_area after thresholding)
        for y in 40..62 {
            for x in 53..75 {
                gray[y as usize * width as usize + x] = 0.55;
            }
        }
        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &DetectConfig::default());

        assert!(result.detections.is_empty());
        assert!(result.stats.rejected_large >= 1);
    }

    #[test]
    fn two_stars_both_detected_flux_ordering() {
        let width = 256u32;
        let height = 256u32;
        let mut gray = flat_background(width, height, 0.15);
        gaussian_bump(width, height, 80.0, 128.0, 0.25, 1.0, &mut gray);
        gaussian_bump(width, height, 180.0, 128.0, 0.35, 1.0, &mut gray);
        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &DetectConfig::default());

        assert_eq!(result.detections.len(), 2);
        assert!(result.detections[0].flux > result.detections[1].flux);
        assert_eq!(result.detections[0].rank, 0);
        assert_eq!(result.detections[1].rank, 1);
    }

    /// Clipped-black denoised sky (phone Night mode / processed JPEG): ~90% of
    /// pixels are exactly 0, MAD = 0. Without the quantization sigma floor the
    /// threshold collapses to ~0 and stars drown in merged JPEG-noise blobs.
    #[test]
    fn clipped_black_sky_recovers_stars_via_sigma_floor() {
        let width = 128u32;
        let height = 128u32;
        let mut gray = vec![0.0f32; (width * height) as usize];
        // Sparse 1-DN noise specks; must stay below the floored threshold.
        for i in (0..gray.len()).step_by(97) {
            gray[i] = 1.0 / 255.0;
        }
        let truth = [(30.2f64, 40.7f64), (80.5, 90.1), (100.3, 30.9)];
        for &(cx, cy) in &truth {
            gaussian_bump(width, height, cx, cy, 0.5, 1.2, &mut gray);
        }

        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &DetectConfig::default());

        assert!(
            result.stats.sigma >= 1.0 / 255.0,
            "sigma not floored: {}",
            result.stats.sigma
        );
        assert_eq!(
            result.detections.len(),
            3,
            "expected exactly the 3 stars, got {:?}",
            result.detections
        );
        for &(cx, cy) in &truth {
            let hit = result
                .detections
                .iter()
                .any(|d| (d.x - cx).abs() < 0.5 && (d.y - cy).abs() < 0.5);
            assert!(hit, "no detection near ({cx},{cy})");
        }
    }

    /// A saturated star blooming past `max_area` must be kept (round + at
    /// peak), while an equally large dim patch stays rejected as extended.
    #[test]
    fn saturated_round_star_above_max_area_kept() {
        let width = 128u32;
        let height = 128u32;
        let mut gray = flat_background(width, height, 0.15);
        // Saturated disk, radius 12 → area ~450 > max_area 150.
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                if (x - 64) * (x - 64) + (y - 64) * (y - 64) <= 12 * 12 {
                    gray[y as usize * width as usize + x as usize] = 1.0;
                }
            }
        }
        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &DetectConfig::default());
        assert_eq!(result.detections.len(), 1, "saturated star kept");
        let det = &result.detections[0];
        assert!(
            (det.x - 64.0).abs() < 1.0 && (det.y - 64.0).abs() < 1.0,
            "centroid at disk center: ({}, {})",
            det.x,
            det.y
        );
        assert!(det.area > 150, "area above point-source cap: {}", det.area);

        // Same footprint at low amplitude: extended object, still rejected.
        let mut dim = flat_background(width, height, 0.15);
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                if (x - 64) * (x - 64) + (y - 64) * (y - 64) <= 12 * 12 {
                    dim[y as usize * width as usize + x as usize] = 0.45;
                }
            }
        }
        let frame = make_frame(width, height, dim);
        let result = detect_stars(&frame, &DetectConfig::default());
        assert!(result.detections.is_empty(), "dim patch must not pass");
        assert!(result.stats.rejected_large >= 1);
    }

    /// Milky-Way star clouds / nebulosity fragments that fit under `max_area`
    /// (the cygnus failure mode: flat blobs of area ~110–150 dominate the
    /// flux-ranked top-N) must be screened by peak concentration. The
    /// statistic is shape-only — a flat disk scores ≈ 2 regardless of
    /// amplitude, a PSF-like peak scores ≫ 2.5.
    #[test]
    fn diffuse_cloud_rejected_compact_star_kept() {
        let width = 128u32;
        let height = 128u32;
        let mut gray = flat_background(width, height, 0.15);
        // Flat dim disk r=4: thresholded blob lands between
        // CONCENTRATION_MIN_AREA and max_area with concentration ≈ 2.
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                if (x - 90) * (x - 90) + (y - 90) * (y - 90) <= 4 * 4 {
                    gray[y as usize * width as usize + x as usize] += 0.1;
                }
            }
        }
        // Compact star elsewhere.
        gaussian_bump(width, height, 32.0, 32.0, 0.5, 1.2, &mut gray);

        let frame = make_frame(width, height, gray);
        let result = detect_stars(&frame, &DetectConfig::default());

        assert!(
            result.stats.rejected_diffuse >= 1,
            "flat blob not screened: {:?}",
            result.stats
        );
        assert_eq!(result.detections.len(), 1, "only the star survives");
        let det = &result.detections[0];
        assert!(
            (det.x - 32.0).abs() < 0.5 && (det.y - 32.0).abs() < 0.5,
            "star centroid: ({}, {})",
            det.x,
            det.y
        );
    }

    #[test]
    fn real_data_smoke_if_present() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let input_dir = repo_root.join("data/input");

        let frames = [
            "2011-09-20_23-49-17-390_Gain-128_Exp-20m.bmp",
            "g128_40ms_1s.bmp",
        ];
        for name in frames {
            let path = input_dir.join(name);
            if !path.exists() {
                continue;
            }
            let frame = FrameImage::load(&path).expect("load frame");
            let result = detect_stars(&frame, &DetectConfig::default());
            assert!(
                (4..=40).contains(&result.stats.accepted),
                "{}: accepted={}",
                name,
                result.stats.accepted
            );
            for det in &result.detections {
                assert!(
                    det.x >= 8.0,
                    "{}: detection x={} < 8 (hot column)",
                    name,
                    det.x
                );
            }
        }

        let blob_path = input_dir.join("g128_40ms_4.bmp");
        if blob_path.exists() {
            let frame = FrameImage::load(&blob_path).expect("load frame");
            let result = detect_stars(&frame, &DetectConfig::default());
            // The frame's big defocused blob must be screened out by one of
            // the extended-object filters (which bucket depends on where the
            // adaptive threshold lands), and never accepted as a star.
            let screened = result.stats.rejected_large + result.stats.rejected_diffuse;
            assert!(
                screened >= 1,
                "g128_40ms_4.bmp: expected the big blob screened (large+diffuse), stats {:?}",
                result.stats
            );
            assert!(
                result
                    .detections
                    .iter()
                    .all(|d| d.area <= DetectConfig::default().max_area),
                "g128_40ms_4.bmp: no oversized blob may pass on this frame"
            );
        }
    }
}
