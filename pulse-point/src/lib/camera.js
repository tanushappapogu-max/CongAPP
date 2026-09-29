// Camera helpers: pick the main (1×) back camera at 1× zoom, toggle torch, capture a JPEG of
// the current frame.
//
// The main camera matters: the ultra-wide (0.5×) lens makes every object half the size in the
// frame, so small things like keys and glasses get missed, and its much wider field of view breaks
// the distance math, which assumes a normal lens.

const NOT_MAIN = /ultra|tele|dual|triple|macro|depth|wide|front|user/i;

/** The main back camera among enumerated devices, or null when labels don't tell us. */
export function pickMainBackCamera(devices) {
  const cameras = devices.filter(d => d.kind === 'videoinput' && d.label);
  const back = cameras.filter(d => /back|rear|environment/i.test(d.label));
  return back.find(d => !NOT_MAIN.test(d.label)) || null;
}

export async function getMainCameraStream() {
  const size = { width: { ideal: 1920 }, height: { ideal: 1080 } };
  const first = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, ...size, aspectRatio: { ideal: 16 / 9 } },
    audio: false,
  });
  // Labels are only readable after permission is granted, so look now and switch if the browser
  // handed us a different back camera (some pick the ultra-wide or a multi-lens virtual camera).
  const main = pickMainBackCamera(await navigator.mediaDevices.enumerateDevices());
  const current = first.getVideoTracks()[0]?.getSettings?.().deviceId;
  if (!main || main.deviceId === current) return first;
  first.getTracks().forEach(t => t.stop());
  try {
    return await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: main.deviceId }, ...size }, audio: false });
  } catch {
    return navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, ...size }, audio: false });
  }
}

/** Set 1× zoom where the camera supports zoom (multi-lens cameras can start at 0.5×). */
export async function setNormalZoom(stream) {
  const track = stream?.getVideoTracks()[0];
  const zoom = track?.getCapabilities?.()?.zoom;
  if (!zoom) return;
  try {
    await track.applyConstraints({ advanced: [{ zoom: Math.min(zoom.max, Math.max(zoom.min, 1)) }] });
  } catch {
    // some platforms reject mid-stream zoom — fine to ignore
  }
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
