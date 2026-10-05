// URL switches for diagnosing phone memory issues:
//   ?nodepth=1      no depth model
//   ?cpu=1          skip WebGPU
//   ?gpu=1          use WebGPU even where it is off by default (Safari / iOS)
//   ?threads=N      force the CPU backend's thread count (1 = single-threaded)
//   ?depthsize=392  run depth on a smaller square crop (less memory, ~6% noisier)
//   ?server=URL     use this vision server (e.g. http://localhost:8788); ?server=off disables it
//   ?nodetect=1     camera + overlay only; no detection model is loaded or run
//   ?captureonly=1  load the detector and copy each frame into its input, but never run it
const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : new URLSearchParams();
const nav = typeof navigator !== 'undefined' ? navigator : null;

/**
 * Safari's engine: every browser on iPhone/iPad (Chrome included) plus desktop Safari.
 * Its WebGPU path leaked ~150 MB/s while scanning on an iPhone and the tab was killed within
 * ~20 s; the same app on the CPU stayed flat around 360 MB, so WebGPU is off there by default.
 */
export function isWebKit(n = nav) {
  if (!n) return false;
  const ua = n.userAgent || '';
  const iOS = /iPhone|iPad|iPod/.test(ua) || (n.platform === 'MacIntel' && n.maxTouchPoints > 1);
  const desktopSafari = /Safari\//.test(ua) && !/Chrome\/|Chromium\/|Edg\/|OPR\/|Firefox\//.test(ua);
  return iOS || desktopSafari;
}

export const FLAGS = {
  noDepth: params.has('nodepth'),
  forceCpu: params.has('cpu') || (isWebKit() && !params.has('gpu')),
  noDetect: params.has('nodetect'),
  captureOnly: params.has('captureonly'),
  threads: Number(params.get('threads')) || null,
  depthSize: params.get('depthsize') === '392' ? 392 : 518,
  server: params.get('server') === 'off' ? null : params.get('server'),
  serverOff: params.get('server') === 'off',
};
