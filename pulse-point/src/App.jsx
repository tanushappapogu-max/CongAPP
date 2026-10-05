import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Camera, ScanLine, Square, Mic, Settings as SettingsIcon } from 'lucide-react';

import { loadPromptPack, resolvePromptTarget, buildPromptSet } from './detection/prompts.js';
import { measureDepth, isDepthBusy, getDepthBackend } from './detection/depth.js';
import { currentDepthMeters, isRoughlyCentered } from './detection/depthSample.js';
import { estimateDistanceMeters, isCloseEnoughForDepth } from './detection/distance.js';
import { loadModel, runInference, preloadModel, isModelReady, getDetectorInfo } from './detection/engine.js';
import { detectWithServer, isServerAvailable } from './detection/server.js';
import { isRemoteUp } from './detection/remote.js';
import { BoxTracker } from './detection/tracker.js';
import { createScannerSession, SCANNER_EVENTS } from './scanner/scannerSession.js';

import { computeGuidance } from './guidance/compute.js';
import { Haptics } from './guidance/haptics.js';
import { Speaker } from './guidance/speech.js';

import { getWideCameraStream, setWidestZoom, lensFovDeg, stopStream } from './lib/camera.js';
import { startListening, isVoiceSupported, extractTarget } from './lib/voice.js';
import { loadSettings, saveSettings } from './lib/settings.js';

import Announcer from './ui/Announcer.jsx';
import SettingsSheet from './ui/SettingsSheet.jsx';
import WelcomeOverlay from './ui/WelcomeOverlay.jsx';


const ADAPTIVE_FPS_MIN = 4;
const ADAPTIVE_FPS_MAX = 15;
const ADAPTIVE_FPS_INITIAL = 10;
const ADAPTIVE_SLACK_MS = 25;
const HEAVY_COOLDOWN_MS = 2500;
const REMOTE_IN_FLIGHT = 2;
const REMOTE_MIN_INTERVAL_MS = 66; // with the vision server, round trips (not this) set the pace
// GPU depth takes ~0.1–0.5 s, CPU depth several seconds; don't queue work faster than it finishes.
// GPU depth takes ~0.1–0.5 s; CPU depth takes seconds and competes with detection for the CPU.
const DEPTH_INTERVAL_MS = { webgpu: 500, server: 1000, wasm: 4000 };

const SENSITIVITY_PROFILES = {
  gentle: { hapticGap: 720, announceGap: 1700 },
  medium: { hapticGap: 520, announceGap: 1200 },
  sharp:  { hapticGap: 360, announceGap: 800 },
};

const URGENT_SIGNALS = new Set(['reach', 'closer', 'lost']);

export default function App() {
  const videoRef          = useRef(null);
  const canvasRef         = useRef(null);
  const streamRef         = useRef(null);
  const mountedRef        = useRef(true);
  const lastLightRunRef   = useRef(0);
  const lastPredsRef      = useRef([]);
  const freshPredsRef     = useRef(false);
  const inferInFlightRef  = useRef(0);
  const inferSeqRef       = useRef(0);
  const appliedSeqRef     = useRef(0);
  const prevAreaRef       = useRef(0);
  const foundOnceRef      = useRef(false);
  const localTargetRef    = useRef(null);
  const lensFovRef        = useRef(null);
  const depthReadingRef   = useRef(null);
  const lastDepthRunRef   = useRef(0);
  const targetRef         = useRef('');
  const isRunningRef      = useRef(false);
  const settingsRef       = useRef(null);
  const announcementRef   = useRef('');
  const lastAnnouncedSignalRef = useRef('');
  const lastAnnouncedTimeRef   = useRef(0);
  const frameRef          = useRef({ width: 640, height: 480 });
  const guidanceTimeRef   = useRef(0);
  const sessionRef        = useRef(null);
  const lastHeavyRunRef   = useRef(0);
  const aiBoxRef          = useRef(null);
  const aiInFlightRef     = useRef(false);


  const lightFpsRef       = useRef(ADAPTIVE_FPS_INITIAL);
  const inferenceWindowRef = useRef([]);

  const trackerRef = useRef(null);
  if (!trackerRef.current) trackerRef.current = new BoxTracker();

  const hapticsRef = useRef(null);
  if (!hapticsRef.current) hapticsRef.current = new Haptics();
  const speakerRef = useRef(null);
  if (!speakerRef.current) speakerRef.current = new Speaker();

  const [target,        setTargetState]   = useState('');
  const [draftTarget,   setDraftTarget]   = useState('');
  const [status,        setStatus]        = useState('idle');
  const [signal,        setSignal]        = useState('looking');
  const [match,         setMatch]         = useState(null);
  const [error,         setError]         = useState('');
  const [debugInfo,     setDebugInfo]     = useState(null);
  const [isRunning,     setIsRunning]     = useState(false);
  const [hapticsAvail,  setHapticsAvail]  = useState(true);
  const [isListening,   setIsListening]   = useState(false);
  const [mode,          setMode]          = useState('normal');
  const [settings,      setSettings]      = useState(() => loadSettings());
  const [settingsOpen,  setSettingsOpen]  = useState(false);
  const [announcement,  setAnnouncement]  = useState('');
  const [announcementUrgent, setAnnouncementUrgent] = useState(false);
  const [showWelcome,     setShowWelcome]   = useState(() => {
    if (typeof window === 'undefined') return false;
    try {
      return !window.localStorage.getItem('pulsepoint_onboarded');
    } catch {
      return true;
    }
  });



  settingsRef.current = settings;
  announcementRef.current = announcement;

  const canVoice = useMemo(() => isVoiceSupported(), []);
  const speechAvail = useMemo(() => speakerRef.current.isAvailable(), []);

  if (!sessionRef.current) {
    sessionRef.current = createScannerSession({
      initializeCamera,
      initializeModel,
      detect: detectFrame,
      computeGuidance: calculateGuidance,
      intervalMs: 16,
      onEvent: handleSessionEvent,
      onDetection: handleDetection,
      onGuidance: handleGuidance,
      onOutput: handleSessionOutput,
      stopCamera: stopSessionCamera,
    });
  }

  useEffect(() => {
    setHapticsAvail(hapticsRef.current.isAvailable());
    return () => {
      mountedRef.current = false;
      void sessionRef.current?.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Background preload on mount ─────────────────────────────────────────────
  // Kick off the model and prompt-pack downloads in the background immediately.
  // By the time the user taps Start, the bytes are already in memory (or the SW
  // cache). No progress is shown here — this is silent prefetching.
  useEffect(() => {
    loadPromptPack().then(() => {
      if (targetRef.current && !localTargetRef.current) resolveLocalTarget(targetRef.current);
    }).catch(() => { /* non-fatal; retried when scanning starts */ });
    if (isModelReady()) return; // already warm from a previous session
    preloadModel().catch(() => { /* non-fatal; will retry on first loadModel call */ });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    hapticsRef.current.setEnabled(settings.haptics);
    speakerRef.current.setEnabled(settings.speech);
    speakerRef.current.setRate(settings.speechRate);
    saveSettings(settings);
  }, [settings]);

  // Hidden diagnostics for phone testing: add ?debug=1 to the URL.
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has('debug')) return undefined;
    let lastError = '';
    const onError = (e) => { lastError = String(e.message || e.reason?.message || e.reason || 'error').slice(0, 120); };
    window.addEventListener('error', onError);
    window.addEventListener('unhandledrejection', onError);
    const id = setInterval(() => {
      const times = inferenceWindowRef.current;
      const avgMs = times.length ? Math.round(times.reduce((a, b) => a + b, 0) / times.length) : null;
      const label = localTargetRef.current?.label;
      const best = label ? Math.max(0, ...lastPredsRef.current.filter(p => p.class === label).map(p => p.score)) : null;
      const heap = performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null;
      setDebugInfo({ ...getDetectorInfo(), depth: getDepthBackend(), avgMs, label, best, heap, lastError });
    }, 500);
    return () => {
      clearInterval(id);
      window.removeEventListener('error', onError);
      window.removeEventListener('unhandledrejection', onError);
    };
  }, []);

  function setTarget(t) {
    const nextTarget = t.trim();
    targetRef.current = nextTarget;
    setTargetState(nextTarget);
    setDraftTarget(nextTarget);
    foundOnceRef.current  = false;
    lastLightRunRef.current = 0;
    lastPredsRef.current = [];
    freshPredsRef.current = false;
    localTargetRef.current = null;
    trackerRef.current.reset();
    lastAnnouncedSignalRef.current = '';
    lastHeavyRunRef.current = 0;
    aiBoxRef.current = null;
    aiInFlightRef.current = false;
    depthReadingRef.current = null;
    lastDepthRunRef.current = 0;
    resolveLocalTarget(nextTarget);
    sessionRef.current?.setTarget(nextTarget);
  }

  function dismissWelcome() {
    try {
      window.localStorage.setItem('pulsepoint_onboarded', '1');
    } catch {
      // Private browsing or blocked storage should not prevent using the app.
    }
    setShowWelcome(false);
  }

  function resolveLocalTarget(tgt) {
    if (!tgt) return;
    const resolved = resolvePromptTarget(tgt);
    if (!resolved) return;
    localTargetRef.current = {
      label: resolved.item.name,
      score: resolved.score,
      source: resolved.source,
      widthCm: resolved.item.widthCm,
      prompts: buildPromptSet(resolved.item),
    };
  }

  function submitTypedTarget() {
    const text = draftTarget.trim();
    if (!text) return;
    setError('');
    setTarget(text);
    setMode('normal');
    if (!isRunningRef.current) startScanner(text);
    else setStatus('looking');
  }

  function handleScanClick() {
    if (isRunningRef.current) { stopScanner(); return; }
    if (draftTarget.trim()) { submitTypedTarget(); return; }
    startScanner();
  }

  function handleTypedKeyDown(e) {
    if (e.key === 'Enter') { e.preventDefault(); submitTypedTarget(); }
  }

  function startVoice() {
    if (!canVoice) {
      setError('Voice input is not supported on this device. Type below.');
      return;
    }
    setError('');
    setIsListening(true);
    if (hapticsRef.current.isAvailable()) navigator.vibrate?.([80]);
    startListening({
      onResult: handleVoice,
      onError: (err) => {
        setIsListening(false);
        if (err === 'not-allowed' || err === 'service-not-allowed') {
          setError('Microphone blocked. Allow microphone access in browser settings.');
        } else if (err === 'no-speech') {
          setError('No speech heard. Tap to retry.');
        } else if (err === 'network') {
          setError('Voice service unavailable. Check connection or type below.');
        } else if (err === 'not-supported') {
          setError('Voice input is not supported in this browser. Type below.');
        } else {
          setError('Voice input failed. Type below.');
        }
      },
      onEnd: () => setIsListening(false),
    });
  }

  function handleVoice(text) {
    setIsListening(false);
    const extracted = extractTarget(text) || text.trim();
    if (extracted) {
      setTarget(extracted);
      setMode('normal');
      if (!isRunningRef.current) startScanner(extracted);
      else setStatus('looking');
    }
  }

  function resetDetectionState() {
    prevAreaRef.current     = 0;
    foundOnceRef.current    = false;
    lastLightRunRef.current = 0;
    lastPredsRef.current    = [];
    trackerRef.current.reset();
    lastAnnouncedSignalRef.current = '';
    lastHeavyRunRef.current = 0;
    aiBoxRef.current = null;
    aiInFlightRef.current = false;
    depthReadingRef.current = null;
    lastDepthRunRef.current = 0;
  }

  function startScanner(startTarget = targetRef.current) {
    if (isRunningRef.current) return;
    const requestedTarget = typeof startTarget === 'string' ? startTarget.trim() : targetRef.current;
    void sessionRef.current?.start(requestedTarget);
  }

  function stopScanner() {
    sessionRef.current?.stop();
  }

  function recordInferenceTime(ms) {
    const w = inferenceWindowRef.current;
    w.push(ms);
    if (w.length > 5) w.shift();
    const avg = w.reduce((s, n) => s + n, 0) / w.length;
    const targetFps = 1000 / (avg + ADAPTIVE_SLACK_MS);
    lightFpsRef.current = Math.max(ADAPTIVE_FPS_MIN, Math.min(ADAPTIVE_FPS_MAX, targetFps));
  }

  async function detectFrame({ model, target: sessionTarget, signal }) {
    if (signal.aborted) return null;

    const video = videoRef.current;
    if (!video || !model || video.readyState < 2) return null;

    const tgt = sessionTarget || targetRef.current;
    if (tgt && !localTargetRef.current) resolveLocalTarget(tgt);

    const frame = { width: video.videoWidth || 640, height: video.videoHeight || 480, fovDeg: lensFovRef.current };
    frameRef.current = frame;
    const now = performance.now();
    guidanceTimeRef.current = now;

    // Inference runs in the background so the overlay and guidance update every frame instead of
    // waiting on it; that matters most with the vision server, where a round trip is ~250 ms+.
    // With the server, two requests overlap to hide network latency; a reply older than one
    // already applied is dropped.
    const remote = isRemoteUp();
    const lightInterval = remote ? REMOTE_MIN_INTERVAL_MS : 1000 / lightFpsRef.current;
    const maxInFlight = remote ? REMOTE_IN_FLIGHT : 1;
    if (inferInFlightRef.current < maxInFlight && now - lastLightRunRef.current >= lightInterval) {
      lastLightRunRef.current = now;
      inferInFlightRef.current += 1;
      const seq = ++inferSeqRef.current;
      const t0 = performance.now();
      runInference(video, localTargetRef.current?.prompts ?? null)
        .then(preds => {
          if (signal.aborted || seq < appliedSeqRef.current) return;
          appliedSeqRef.current = seq;
          lastPredsRef.current = Array.isArray(preds) ? preds : [];
          freshPredsRef.current = true;
          recordInferenceTime(performance.now() - t0);
        })
        .catch(e => {
          // eslint-disable-next-line no-console
          console.warn('Inference error', e);
        })
        .finally(() => {
          inferInFlightRef.current -= 1;
        });
    }

    // True on the first frame after new detections arrive: the tracker and depth only act on fresh results.
    const ranLight = freshPredsRef.current;
    freshPredsRef.current = false;
    const predictions = lastPredsRef.current;

    if (signal.aborted) return null;

    // ── Optional server probe every HEAVY_COOLDOWN_MS ──
    // Runs whenever configured/healthy, even after local matches; the merge below only uses it
    // when local matching has no result. The camera frame and target are uploaded by server.js.
    const ranHeavy = now - lastHeavyRunRef.current >= HEAVY_COOLDOWN_MS;
    if (ranHeavy && tgt && !aiInFlightRef.current && isServerAvailable()) {
      lastHeavyRunRef.current = now;
      aiInFlightRef.current = true;
      detectWithServer(video, tgt).then(result => {
        aiInFlightRef.current = false;
        if (signal.aborted || !mountedRef.current) return;
        if (result) {
          aiBoxRef.current = result;
        } else {
          aiBoxRef.current = null;
        }
      });
    }

    const localInfo = localTargetRef.current;
    const mappedLabel = localInfo?.label;
    const priorTrack = trackerRef.current.predict(now);
    const localMatchRaw = mappedLabel ? findTarget(predictions, mappedLabel, priorTrack?.bbox, frame) : null;
    const localMatch = localMatchRaw ? {
      ...localMatchRaw,
      displayClass: mappedLabel !== tgt ? tgt : localMatchRaw.class,
      source: localMatchRaw,
    } : null;

    // ── Merge: use the server result when on-device detection has no match ──
    const serverResult = aiBoxRef.current;
    const freshMatch = localMatch || (tgt && serverResult ? {
      ...serverResult,
      displayClass: tgt,
      fromServer: true,
    } : null);

    const depthBackend = getDepthBackend();
    const depthFast = depthBackend === 'webgpu' || depthBackend === 'server';
    const depthInterval = DEPTH_INTERVAL_MS[depthBackend] ?? DEPTH_INTERVAL_MS.wasm;
    if (
      freshMatch && ranLight && now - lastDepthRunRef.current >= depthInterval && !isDepthBusy()
      && (depthFast || isRoughlyCentered(freshMatch.bbox, frame))
      && isCloseEnoughForDepth(
        estimateDistanceMeters(null, freshMatch.bbox[2], Math.max(frame.width, frame.height), localInfo?.widthCm ?? null, frame.fovDeg),
        (freshMatch.bbox[2] * freshMatch.bbox[3]) / (frame.width * frame.height),
      )
    ) {
      lastDepthRunRef.current = now;
      const depthBox = freshMatch.bbox;
      measureDepth(video, depthBox).then(meters => {
        if (meters == null || signal.aborted || !mountedRef.current || targetRef.current !== tgt) return;
        depthReadingRef.current = { meters, boxWidth: depthBox[2], at: performance.now(), target: tgt };
      });
    }

    if (freshMatch && ranLight) {
      trackerRef.current.update(
        freshMatch.bbox,
        freshMatch.score,
        freshMatch.displayClass || freshMatch.class,
        now,
        false,
      );
    }

    const predicted = trackerRef.current.predict(now);
    const displayMatch = predicted ? {
      class: predicted.label,
      displayClass: predicted.label,
      bbox: predicted.bbox,
      score: predicted.confidence,
      fromAi: false,
      ageMs: predicted.ageMs,
      refWidthCm: localTargetRef.current?.widthCm ?? null,
      depthMeters: currentDepthMeters(depthReadingRef.current, tgt, predicted.bbox[2], now),
    } : null;

    draw(predictions, displayMatch);
    return displayMatch;
  }

  function calculateGuidance(detection) {
    const guidance = computeGuidance(detection, frameRef.current, prevAreaRef.current);
    prevAreaRef.current = guidance.area;
    return guidance;
  }

  async function initializeCamera({ target: requestedTarget, signal }) {
    let stream = null;
    try {
      stream = await getWideCameraStream();
      if (signal.aborted) return stream;

      const video = videoRef.current;
      if (!video) throw new Error('Camera preview is unavailable.');
      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      if (signal.aborted) return stream;
      await setWidestZoom(stream);
      lensFovRef.current = lensFovDeg(stream);
      if (signal.aborted) return stream;

      if (mountedRef.current) {
        hapticsRef.current.fire('looking', true);
      }
      return stream;
    } catch (error) {
      if (stream) stopSessionCamera(stream);
      throw error;
    }
  }

  async function initializeModel({ target: requestedTarget, signal }) {
    const [model] = await Promise.all([loadModel({ signal }), loadPromptPack({ signal })]);
    if (requestedTarget && !localTargetRef.current) resolveLocalTarget(requestedTarget);

    if (!signal.aborted && mountedRef.current) {
      setStatus('looking');
      setAnnouncement(requestedTarget ? `Looking for ${requestedTarget}.` : 'Camera active. Say or type a target.');
      speakerRef.current.say('Sixth Sense ready.', { urgent: true });
    }
    return model;
  }

  function stopSessionCamera(stream) {
    if (!stream) return;
    if (streamRef.current === stream) streamRef.current = null;
    const video = videoRef.current;
    if (video?.srcObject === stream) {
      video.pause?.();
      video.srcObject = null;
    }
    stopStream(stream);
  }

  function handleSessionEvent(event) {
    if (event.type === SCANNER_EVENTS.STOP) {
      hapticsRef.current.cancel();
      speakerRef.current.cancel();
      isRunningRef.current = false;
      if (!mountedRef.current) return;
      setIsRunning(false);
      setStatus('idle');
      setMatch(null);
      setSignal('looking');
      resetDetectionState();
      return;
    }

    if (!mountedRef.current) return;

    switch (event.type) {
      case SCANNER_EVENTS.START:
        isRunningRef.current = true;
        setError('');
        setIsRunning(true);
        setStatus('booting');
        setMatch(null);
        setSignal('looking');
        setAnnouncement('Starting camera…');
        resetDetectionState();
        break;
      case SCANNER_EVENTS.TARGET_SET:
        if (isRunningRef.current) {
          setMatch(null);
          setStatus('looking');
          setSignal('looking');
        }
        break;
      case SCANNER_EVENTS.CAMERA_ERROR:
        isRunningRef.current = false;
        setIsRunning(false);
        setStatus('blocked');
        setError(event.error?.message || 'Camera blocked. Allow camera access and try again.');
        break;
      case SCANNER_EVENTS.MODEL_ERROR:
        isRunningRef.current = false;
        setIsRunning(false);
        setStatus('blocked');
        setError(event.error?.message || 'CNN model failed to load. Refresh and try again.');
        break;
      default:
        break;
    }
  }

  function handleDetection(detection, event) {
    if (!mountedRef.current) return;
    const displayMatch = detection;
    const guidance = event.guidance;
    setMatch({
      name: displayMatch.displayClass || displayMatch.class,
      score: displayMatch.score,
      direction: guidance?.direction,
      distance: guidance?.distance,
      distanceMeters: guidance?.distanceMeters ?? null,
      fromAi: false,
    });
  }

  function handleGuidance(guidance) {
    if (!mountedRef.current) return;
    const now = guidanceTimeRef.current || performance.now();
    setStatus(guidance.status);
    setSignal(guidance.signal);

    if (!foundOnceRef.current) {
      foundOnceRef.current = true;
      hapticsRef.current.fire('found', true);
      setAnnouncement(guidance.sentence);
      setAnnouncementUrgent(true);
      speakerRef.current.say(guidance.speechPhrase, { urgent: true, force: true });
      lastAnnouncedSignalRef.current = guidance.signal;
      lastAnnouncedTimeRef.current = now;
    } else {
      hapticsRef.current.fire(guidance.signal);
      maybeAnnounce(guidance, now);
    }
  }

  function handleSessionOutput(event) {
    if (event.type !== SCANNER_EVENTS.DETECTION_MISSED) return;
    draw(lastPredsRef.current, null);
    if (!mountedRef.current) return;

    setMatch(null);
    setSignal(event.lost ? 'lost' : 'looking');
    setStatus(event.lost ? 'lost' : 'looking');
    prevAreaRef.current = 0;
    if (event.target) {
      if (event.lost) {
        foundOnceRef.current = false;
        hapticsRef.current.fire('lost');
      } else {
        hapticsRef.current.fire('looking');
      }
      throttledAnnounce(event.lost ? `Lost ${event.target}. Looking again.` : `Looking for ${event.target}.`, event.lost);
    }
  }

  function maybeAnnounce(g, now) {
    const profile = SENSITIVITY_PROFILES[settingsRef.current.sensitivity] || SENSITIVITY_PROFILES.medium;
    const signalChanged = g.signal !== lastAnnouncedSignalRef.current;
    const timeOk = now - lastAnnouncedTimeRef.current >= profile.announceGap;
    
    // Only announce if: truly urgent, or enough time has passed (not on every direction change)
    if (!URGENT_SIGNALS.has(g.signal) && !timeOk) return;
    
    if (URGENT_SIGNALS.has(g.signal) || signalChanged || timeOk) {
      lastAnnouncedSignalRef.current = g.signal;
      lastAnnouncedTimeRef.current = now;
      setAnnouncement(g.sentence);
      setAnnouncementUrgent(URGENT_SIGNALS.has(g.signal) || signalChanged);
      
      // Only force interrupt for truly urgent signals (reach, closer)
      const shouldForce = URGENT_SIGNALS.has(g.signal);
      speakerRef.current.say(g.speechPhrase, { 
        urgent: URGENT_SIGNALS.has(g.signal),
        force: shouldForce
      });
    }
  }

  function throttledAnnounce(text, urgent) {
    const profile = SENSITIVITY_PROFILES[settingsRef.current.sensitivity] || SENSITIVITY_PROFILES.medium;
    const now = performance.now();
    if (text === announcementRef.current && now - lastAnnouncedTimeRef.current < profile.announceGap) return;
    lastAnnouncedTimeRef.current = now;
    setAnnouncement(text);
    setAnnouncementUrgent(urgent);
  }

  function draw(predictions, targetMatch) {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const W = video.videoWidth || 640, H = video.videoHeight || 480;
    const displayW = canvas.clientWidth  || W;
    const displayH = canvas.clientHeight || H;
    // Resizing reallocates the canvas; doing it every tick churns memory on phones.
    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width  = displayW;
      canvas.height = displayH;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, displayW, displayH);

    const scale   = Math.max(displayW / W, displayH / H);
    const offsetX = (W * scale - displayW) / 2;
    const offsetY = (H * scale - displayH) / 2;
    const mapBox  = ([x, y, w, h]) => [x * scale - offsetX, y * scale - offsetY, w * scale, h * scale];

    // ── Feature extraction grid overlay (7×7 anchor grid) ──
    if (isRunningRef.current) {
      const GX = 7, GY = 7;
      const cw = displayW / GX, ch = displayH / GY;
      ctx.strokeStyle = 'rgba(0,255,157,0.045)';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([]);
      for (let r = 0; r <= GY; r++) {
        ctx.beginPath(); ctx.moveTo(0, r * ch); ctx.lineTo(displayW, r * ch); ctx.stroke();
      }
      for (let c = 0; c <= GX; c++) {
        ctx.beginPath(); ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, displayH); ctx.stroke();
      }
    }

    // ── All YOLO boxes (showAllBoxes mode) ──
    if (settingsRef.current.showAllBoxes) {
      predictions.forEach(p => {
        if (targetMatch?.source === p) return;
        const [x, y, w, h] = mapBox(p.bbox);
        ctx.strokeStyle = 'rgba(255,255,255,0.20)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
      });
    }

    if (!targetMatch) return;

    const [x, y, bw, bh] = mapBox(targetMatch.bbox);
    const cx = x + bw / 2, cy = y + bh / 2;
    const opacity = targetMatch.ageMs > 120 ? 0.72 : 1.0;
    ctx.globalAlpha = opacity;

    // ── Attention heatmap behind box ──
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(bw, bh) * 0.75);
    grad.addColorStop(0, 'rgba(0,255,157,0.10)');
    grad.addColorStop(0.5, 'rgba(0,255,157,0.04)');
    grad.addColorStop(1,   'rgba(0,255,157,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(x - bw * 0.25, y - bh * 0.25, bw * 1.5, bh * 1.5);

    // ── Anchor lines from nearest grid intersections to box center ──
    const GX = 7, GY = 7;
    const cw = displayW / GX, ch = displayH / GY;
    const nearCol = Math.round(cx / cw);
    const nearRow = Math.round(cy / ch);
    ctx.setLineDash([2, 5]);
    ctx.lineWidth = 0.8;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const gx = (nearCol + dc) * cw;
        const gy = (nearRow + dr) * ch;
        if (gx < 0 || gx > displayW || gy < 0 || gy > displayH) continue;
        const dist = Math.hypot(gx - cx, gy - cy);
        ctx.strokeStyle = `rgba(0,255,157,${Math.max(0, 0.18 - dist / (displayW * 0.6))})`;
        ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(cx, cy); ctx.stroke();
        // Anchor dot
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(gx, gy, 2, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,255,157,0.28)'; ctx.fill();
        ctx.setLineDash([2, 5]);
      }
    }
    ctx.setLineDash([]);

    // ── Corner bracket box (technical CNN style) ──
    const cl = Math.min(bw, bh) * 0.22;
    ctx.strokeStyle = '#00ff9d';
    ctx.lineWidth = 2.5;
    // TL
    ctx.beginPath(); ctx.moveTo(x, y + cl); ctx.lineTo(x, y); ctx.lineTo(x + cl, y); ctx.stroke();
    // TR
    ctx.beginPath(); ctx.moveTo(x + bw - cl, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + cl); ctx.stroke();
    // BL
    ctx.beginPath(); ctx.moveTo(x, y + bh - cl); ctx.lineTo(x, y + bh); ctx.lineTo(x + cl, y + bh); ctx.stroke();
    // BR
    ctx.beginPath(); ctx.moveTo(x + bw - cl, y + bh); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw, y + bh - cl); ctx.stroke();

    // Faint full outline
    ctx.strokeStyle = 'rgba(0,255,157,0.25)';
    ctx.lineWidth = 0.8;
    ctx.strokeRect(x, y, bw, bh);

    // Center crosshair
    const cl2 = 7;
    ctx.strokeStyle = 'rgba(0,255,157,0.55)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - cl2, cy); ctx.lineTo(cx + cl2, cy);
    ctx.moveTo(cx, cy - cl2); ctx.lineTo(cx, cy + cl2);
    ctx.stroke();

    ctx.globalAlpha = 1;

    // ── Label chip (monospace, technical) ──
    const display = targetMatch.displayClass || targetMatch.class;
    const conf    = Math.round(targetMatch.score * 100);
    const label   = `${display.toUpperCase()}  ${conf}%`;
    ctx.font = '600 12px "JetBrains Mono", ui-monospace, monospace';
    const tw = ctx.measureText(label).width;
    const lw = tw + 18, lh = 20;
    const ly = Math.max(lh + 2, y) - lh - 2;

    ctx.fillStyle = '#00ff9d';
    ctx.beginPath();
    ctx.roundRect(x, ly, lw, lh, 3);
    ctx.fill();

    ctx.fillStyle = '#021108';
    ctx.fillText(label, x + 9, ly + 14);

    // Pixel coord annotation (bottom-right of box)
    ctx.font = '500 9px ui-monospace, monospace';
    ctx.fillStyle = 'rgba(0,255,157,0.45)';
    const coordTxt = `[${Math.round(x)},${Math.round(y)}]`;
    ctx.fillText(coordTxt, x + 3, Math.min(displayH - 4, y + bh + 11));
  }

  return (
    <>
      <a href="#target-input" className="sr-only skip-link">Skip to search</a>
      {showWelcome && <WelcomeOverlay onDismiss={dismissWelcome} />}

      <main className={`scanner signal-${signal}`}>
      <h1 className="sr-only">Sixth Sense Object Finder</h1>
      <video ref={videoRef} playsInline muted aria-hidden="true" />
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={match ? `Detection overlay: ${match.name} ${match.direction}, ${match.distance}` : 'Detection overlay'}
      />

      <Announcer message={announcement} urgent={announcementUrgent} />

      {debugInfo && (
        <pre className="debug-readout" aria-hidden="true">
          {`server ${debugInfo.server ?? '-'} · yoloe ${debugInfo.yoloe ?? '-'} · yolo ${debugInfo.yolo ?? '-'} · depth ${debugInfo.depth ?? '-'} · threads ${debugInfo.threads ?? '-'}
frame ${debugInfo.avgMs ?? '-'} ms · target ${debugInfo.label ?? '-'} · best ${debugInfo.best == null ? '-' : debugInfo.best.toFixed(2)}
heap ${debugInfo.heap ?? '-'} MB${debugInfo.lastError ? `\nerr ${debugInfo.lastError}` : ''}`}
        </pre>
      )}

      {!isRunning && (
        <button className="start-button" type="button" onClick={startScanner} aria-label="Start CNN scanner">
          <Camera size={30} aria-hidden="true" />
          <span>Start</span>
        </button>
      )}

      {/* Top-right control rail */}
      <div className="top-rail" role="group" aria-label="Quick controls">
        <button
          type="button"
          className="rail-btn"
          onClick={() => setSettingsOpen(true)}
          aria-label="Open settings"
        >
          <SettingsIcon size={18} aria-hidden="true" />
        </button>
      </div>

      {error && (
        <div className="error-pill" role="alert">
          {error}
          <button type="button" className="error-dismiss" onClick={() => setError('')} aria-label="Dismiss error">×</button>
        </div>
      )}

      <div className="target-bar">
        <button
          className={`mic-btn${isListening ? ' mic-active' : ''}${!canVoice ? ' mic-disabled' : ''}`}
          type="button"
          onClick={startVoice}
          aria-label={isListening ? 'Listening' : canVoice ? 'Tap to speak' : 'Voice not supported, type instead'}
          disabled={!canVoice}
        >
          <Mic size={22} aria-hidden="true" />
        </button>

        <div className="target-display" aria-label="Target object">
          <input
            className="target-input"
            id="target-input"
            type="text"
            value={draftTarget}
            onChange={e => setDraftTarget(e.target.value)}
            onKeyDown={handleTypedKeyDown}
            placeholder="say or type what to find…"
            autoComplete="off"
            autoCapitalize="off"
            enterKeyHint="go"
            aria-label="Type a target to find"
          />
          <button className="go-btn" type="button" onClick={submitTypedTarget} aria-label="Submit target">
            Go
          </button>
        </div>

        <button
          type="button"
          className="scan-btn"
          onClick={handleScanClick}
          aria-label={isRunning ? 'Stop scanning' : 'Start scanning'}
        >
          {isRunning ? <Square size={18} aria-hidden="true" /> : <ScanLine size={20} aria-hidden="true" />}
        </button>
      </div>

      <SettingsSheet
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={setSettings}
        hapticsAvailable={hapticsAvail}
        speechAvailable={speechAvail}
      />
      </main>
    </>
  );
}

function findTarget(predictions, target, priorBox = null, frame = null) {
  return predictions
    .filter(p => p.class === target)
    .sort((a, b) => {
      if (!priorBox || !frame) return b.score - a.score;

      const continuity = box => {
        const [x, y, w, h] = box;
        const [px, py, pw, ph] = priorBox;
        const centerDistance = Math.hypot(
          ((x + w / 2) - (px + pw / 2)) / frame.width,
          ((y + h / 2) - (py + ph / 2)) / frame.height,
        );
        const x1 = Math.max(x, px);
        const y1 = Math.max(y, py);
        const x2 = Math.min(x + w, px + pw);
        const y2 = Math.min(y + h, py + ph);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const union = w * h + pw * ph - inter;
        const iou = union <= 0 ? 0 : inter / union;
        return iou * 0.65 + Math.max(0, 1 - centerDistance * 4) * 0.35;
      };

      return (b.score + continuity(b.bbox) * 1.8) - (a.score + continuity(a.bbox) * 1.8);
    })[0];
}
