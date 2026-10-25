// Pure helpers for turning a metric depth map into one distance for the target box.

const ROI_SHRINK = 0.2;       // ignore the outer 20% of the box on each side (mostly background)
const FOREGROUND_PCTL = 0.3;  // objects sit in front of their background, so bias toward nearer pixels
const MAX_DEPTH_AGE_MS = 15000; // CPU readings take seconds; the box-width ratio keeps them current
const MIN_METERS = 0.1;
const MAX_METERS = 20;

/**
 * @param {Float32Array} depth  row-major depth map in meters covering the full frame
 * @param {number} dw  depth map width
 * @param {number} dh  depth map height
 * @param {[number, number, number, number]} boxRel  [x, y, w, h] as fractions of the frame
 * @returns {number|null} meters
 */
export function sampleBoxDepth(depth, dw, dh, boxRel) {
  const [x, y, w, h] = boxRel;
  const x0 = Math.max(0, Math.floor((x + w * ROI_SHRINK) * dw));
  const x1 = Math.min(dw, Math.ceil((x + w * (1 - ROI_SHRINK)) * dw));
  const y0 = Math.max(0, Math.floor((y + h * ROI_SHRINK) * dh));
  const y1 = Math.min(dh, Math.ceil((y + h * (1 - ROI_SHRINK)) * dh));
  if (x1 <= x0 || y1 <= y0) return null;

  const values = [];
  for (let row = y0; row < y1; row++) {
    for (let col = x0; col < x1; col++) {
      const v = depth[row * dw + col];
      if (Number.isFinite(v) && v > 0) values.push(v);
    }
  }
  if (!values.length) return null;
  values.sort((a, b) => a - b);
  const meters = values[Math.min(values.length - 1, Math.floor(values.length * FOREGROUND_PCTL))];
  return Math.min(MAX_METERS, Math.max(MIN_METERS, meters));
}

/**
 * The largest centered square in the frame. Depth runs on this crop instead of the whole wide
 * frame: on 21 indoor photos it read within ~2% of the full frame (after a +3% correction) at
 * about half the peak memory, because the model compares every image patch with every other.
 */
export function centerSquare(frameWidth, frameHeight) {
  const side = Math.min(frameWidth, frameHeight);
  return { left: (frameWidth - side) / 2, top: (frameHeight - side) / 2, side };
}

/** The box as fractions of the crop, or null when its center falls outside the crop. */
export function boxInCrop(bbox, crop) {
  const [x, y, w, h] = bbox;
  const cx = x + w / 2 - crop.left;
  const cy = y + h / 2 - crop.top;
  if (cx < 0 || cy < 0 || cx > crop.side || cy > crop.side) return null;
  return [(x - crop.left) / crop.side, (y - crop.top) / crop.side, w / crop.side, h / crop.side];
}

/**
 * True when the box center sits in the middle half of the frame, i.e. the user is roughly facing
 * the target. Slow (CPU) depth only runs then: while turning toward it, direction is all that matters.
 */
export function isRoughlyCentered(bbox, frame) {
  const [x, y, w, h] = bbox;
  const cx = (x + w / 2) / frame.width;
  const cy = (y + h / 2) / frame.height;
  return cx >= 0.25 && cx <= 0.75 && cy >= 0.25 && cy <= 0.75;
}

/**
 * Carry a depth reading forward: if the box is now twice as wide, the object is half as far.
 * @param {{ meters: number, boxWidth: number, at: number, target: string } | null} reading
 * @returns {number|null}
 */
export function currentDepthMeters(reading, target, boxWidth, now) {
  if (!reading || reading.target !== target) return null;
  if (!(boxWidth > 0) || !(reading.boxWidth > 0)) return null;
  if (now - reading.at > MAX_DEPTH_AGE_MS) return null;
  const meters = reading.meters * (reading.boxWidth / boxWidth);
  return Math.min(MAX_METERS, Math.max(MIN_METERS, meters));
}
