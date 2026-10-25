// Camera helpers: pick the widest back camera at its widest zoom, report that lens's field of
// view for the distance math, toggle torch, capture a JPEG of the current frame.
//
// The wide lens sees more of the room, so the user finds the target without sweeping as far. Its
// field of view is much wider than a normal lens, so distance math must use lensFovDeg(), not the
// main camera's ~64°.

const ULTRA = /ultra/i;
const VIRTUAL = /dual|triple/i;
const NOT_MAIN = /ultra|tele|dual|triple|macro|depth|wide|front|user/i;
const MAIN_LONG_SIDE_FOV_DEG = 64;

/** The main back camera among enumerated devices, or null when labels don't tell us. */
export function pickMainBackCamera(devices) {
  const cameras = devices.filter(d => d.kind === 'videoinput' && d.label);
  const back = cameras.filter(d => /back|rear|environment/i.test(d.label));
  return back.find(d => !NOT_MAIN.test(d.label)) || null;
}

/**
 * The dedicated ultra-wide back camera if there is one (not a multi-lens virtual camera, whose zoom
 * scale we can't read reliably), else the main back camera.
 */
export function pickWideBackCamera(devices) {
  const cameras = devices.filter(d => d.kind === 'videoinput' && d.label);
  const back = cameras.filter(d => /back|rear|environment/i.test(d.label));
  return back.find(d => ULTRA.test(d.label) && !VIRTUAL.test(d.label)) || pickMainBackCamera(devices);
}

export async function getWideCameraStream() {
  const size = { width: { ideal: 1920 }, height: { ideal: 1080 } };
  const first = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, ...size, aspectRatio: { ideal: 16 / 9 } },
    audio: false,
  });
  // Labels are only readable after permission is granted, so look now and switch lenses if needed.
  const wide = pickWideBackCamera(await navigator.mediaDevices.enumerateDevices());
  const current = first.getVideoTracks()[0]?.getSettings?.().deviceId;
  if (!wide || wide.deviceId === current) return first;
  first.getTracks().forEach(t => t.stop());
  try {
    return await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: wide.deviceId }, ...size }, audio: false });
  } catch {
    return navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, ...size }, audio: false });
  }
}

/** Zoom all the way out where the camera supports zoom (some Android main cameras reach ~0.6×). */
export async function setWidestZoom(stream) {
  const track = stream?.getVideoTracks()[0];
  const zoom = track?.getCapabilities?.()?.zoom;
  if (!zoom) return;
  try {
    await track.applyConstraints({ advanced: [{ zoom: zoom.min }] });
  } catch {
    // some platforms reject mid-stream zoom — fine to ignore
  }
}

/**
 * Field of view across the frame's long side, in degrees. The main camera is ~64°; the ultra-wide
 * lens counts as 0.5× and zoom scales from there (tan of the half-angle divides by magnification),
 * so the ultra-wide comes out ~103°.
 */
export function lensFovDeg(stream) {
  const track = stream?.getVideoTracks?.()[0];
  if (!track) return MAIN_LONG_SIDE_FOV_DEG;
  const zoom = track.getSettings?.().zoom || 1;
  const magnification = (ULTRA.test(track.label || '') ? 0.5 : 1) * zoom;
  const half = Math.atan(Math.tan((MAIN_LONG_SIDE_FOV_DEG / 2) * Math.PI / 180) / magnification);
  return Math.min(130, Math.max(20, (2 * half * 180) / Math.PI));
}

export function hasTorchSupport(stream) {
  const track = stream?.getVideoTracks()[0];
  return Boolean(track?.getCapabilities?.()?.torch);
}

export async function setTorch(stream, on) {
  const track = stream?.getVideoTracks()[0];
  if (!track) return false;
  try {
    await track.applyConstraints({ advanced: [{ torch: !!on }] });
    return true;
  } catch {
    return false;
  }
}

export function captureJpeg(video, quality = 0.82) {
  if (!video) return null;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', quality);
}

export function stopStream(stream) {
  stream?.getTracks().forEach(t => t.stop());
}
