import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Camera, ScanLine, Square, Mic, Settings as SettingsIcon, Flashlight, FlashlightOff } from 'lucide-react';

import { resolveCocoTarget, findClosestCocoLabel, TARGET_ALIASES, normalizeTargetText } from './detection/coco.js';
import { loadModel, runInference, preloadModel, isModelReady } from './detection/engine.js';
import { detectWithServer, isServerAvailable } from './detection/server.js';
import { BoxTracker } from './detection/tracker.js';
import { KNOWN_OBJECTS } from './detection/objectList.js';
import { createScannerSession, SCANNER_EVENTS } from './scanner/scannerSession.js';

import { computeGuidance } from './guidance/compute.js';
import { Haptics } from './guidance/haptics.js';
import { Speaker } from './guidance/speech.js';

import { getWideCameraStream, setWidestZoom, hasTorchSupport, setTorch, stopStream } from './lib/camera.js';
import { startListening, isVoiceSupported, extractTarget } from './lib/voice.js';
import { loadSettings, saveSettings } from './lib/settings.js';

import Announcer from './ui/Announcer.jsx';
import SettingsSheet from './ui/SettingsSheet.jsx';
import FeatureGrid from './ui/FeatureGrid.jsx';
import WelcomeOverlay from './ui/WelcomeOverlay.jsx';
import StartupProgress from './ui/StartupProgress.jsx';
import { callGeminiBox } from './detection/ai.js';


// CNN architecture displayed in the ribbon (YOLOv8n backbone + FPN + head)
const ARCH_LAYERS = [
  { id: 'input', label: 'INPUT',   dim: '3×640' },
  { id: 'c1',    label: 'CONV',    dim: '32×320' },
  { id: 'p1',    label: 'POOL',    dim: '32×160' },
  { id: 'c2',    label: 'CONV',    dim: '64×80'  },
  { id: 'p2',    label: 'POOL',    dim: '64×40'  },
  { id: 'c3',    label: 'CONV',    dim: '128×20' },
  { id: 'fpn',   label: 'FPN',     dim: '3×128'  },
  { id: 'head',  label: 'HEAD',    dim: '8400×85'},
  { id: 'nms',   label: 'NMS',     dim: 'detect' },
];

// Right-side layer depth panel (deep path through backbone)
const LAYER_STACK = [
  { name: 'CONV2D',     dim: '32×320×320', fill: 1.0 },
  { name: 'BATCHNORM',  dim: '32',         fill: 1.0 },
  { name: 'C2F-BLOCK',  dim: '64×160×160', fill: 0.85 },
  { name: 'CONV2D',     dim: '128×80×80',  fill: 0.72 },
  { name: 'C2F-BLOCK',  dim: '128×80×80',  fill: 0.65 },
  { name: 'CONV2D',     dim: '256×40×40',  fill: 0.52 },
  { name: 'C2F-BLOCK',  dim: '256×40×40',  fill: 0.44 },
  { name: 'SPPF',       dim: '512×20×20',  fill: 0.35 },
  { name: 'FPN-UP',     dim: '256×40×40',  fill: 0.28 },
  { name: 'DETECT',     dim: '85×8400',    fill: 0.18 },
];

const ADAPTIVE_FPS_MIN = 4;
const ADAPTIVE_FPS_MAX = 15;
const ADAPTIVE_FPS_INITIAL = 10;
const ADAPTIVE_SLACK_MS = 25;
const HEAVY_COOLDOWN_MS = 2500;
const CLOUD_AI_COOLDOWN_MS = 5000;

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
  const prevAreaRef       = useRef(0);
  const foundOnceRef      = useRef(false);
  const localTargetRef    = useRef(null);
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
  const cloudAiBoxRef     = useRef(null);
  const cloudAiInFlightRef = useRef(false);
  const lastCloudAiRunRef = useRef(0);
  const cloudAiRequestRef = useRef(0);

  const objPanelRef        = useRef(null);
  const objPanelSearchRef  = useRef(null);
  const objPanelTriggerRef = useRef(null);
  const objPanelFocusRef   = useRef(null);

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
  const [isRunning,     setIsRunning]     = useState(false);
  const [hapticsAvail,  setHapticsAvail]  = useState(true);
  const [isListening,   setIsListening]   = useState(false);
  const [mode,          setMode]          = useState('normal');
  const [settings,      setSettings]      = useState(() => loadSettings());
  const [settingsOpen,  setSettingsOpen]  = useState(false);
  const [torchOn,       setTorchOn]       = useState(false);
  const [torchAvail,    setTorchAvail]    = useState(false);
  const [announcement,  setAnnouncement]  = useState('');
  const [announcementUrgent, setAnnouncementUrgent] = useState(false);
  const [cnnMs,          setCnnMs]          = useState(null);
  const [cnnConf,        setCnnConf]        = useState(null);
  const [serverMs,       setServerMs]       = useState(null);
  const [serverLabel,    setServerLabel]    = useState('');
  const [serverModel,    setServerModel]    = useState('');
  const [activeLayerIdx, setActiveLayerIdx] = useState(0);
  const [alternatives,   setAlternatives]   = useState([]);
  const [objPanelOpen,   setObjPanelOpen]   = useState(false);
  const [objFilter,      setObjFilter]      = useState('');
  const [showWelcome,     setShowWelcome]   = useState(() => {
    if (typeof window === 'undefined') return false;
    try {
      return !window.localStorage.getItem('pulsepoint_onboarded');
    } catch {
      return true;
    }
  });

  // ── Boot progress HUD ───────────────────────────────────────────────────────
  // Tracks each startup step so StartupProgress.jsx can render accurate state.
  const BOOT_STEP_IDS = ['camera', 'download', 'compile', 'warmup'];

  function makeSteps() {
    return [
      { id: 'camera',   label: 'Camera sensor',          detail: 'Requesting wide-angle feed…', state: 'pending', subPercent: null },
      { id: 'download', label: 'Neural weights (12.8 MB)', detail: '', state: 'pending', subPercent: null },
      { id: 'compile',  label: 'ONNX graph compilation', detail: 'Allocating WASM SIMD operators…', state: 'pending', subPercent: null },
      { id: 'warmup',   label: 'Engine warm-up',         detail: 'Pre-compiling JIT kernels…', state: 'pending', subPercent: null },
    ];
  }

  const [bootActive,   setBootActive]   = useState(false);
  const [bootSteps,    setBootSteps]    = useState(makeSteps);
  const [bootPct,      setBootPct]      = useState(0);
  const [isFirstBoot,  setIsFirstBoot]  = useState(false);
  const bootAbortRef                    = useRef(null);

  /** Update a single boot step by id. Merges the patch object into the existing step. */
  const patchBootStep = useCallback((id, patch) => {
    setBootSteps(prev => prev.map(s => s.id === id ? { ...s, ...patch } : s));
  }, []);

  /** Recalculate overall progress from step states. */
  function calcBootPct(steps) {
    const weights = { camera: 10, download: 55, compile: 20, warmup: 15 };
    let pct = 0;
    for (const s of steps) {
      const w = weights[s.id] ?? 10;
      if (s.state === 'done')   pct += w;
      else if (s.state === 'active') pct += (w * (s.subPercent ?? 50)) / 100;
    }
    return Math.min(100, Math.round(pct));
  }


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
  // Kick off the 12.8 MB model download in the background immediately. By the
  // time the user taps Start, the bytes are already in memory (or the SW cache).
  // No progress is shown here — this is silent prefetching.
  useEffect(() => {
    if (isModelReady()) return; // already warm from a previous session
    preloadModel().catch(() => { /* non-fatal; will retry on first loadModel call */ });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    hapticsRef.current.setEnabled(settings.haptics);
    speakerRef.current.setEnabled(settings.speech);
    speakerRef.current.setRate(settings.speechRate);
    saveSettings(settings);
  }, [settings]);

  // Cycle active layer in the architecture ribbon during inference
  useEffect(() => {
    if (!isRunning) { setActiveLayerIdx(0); return; }
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setActiveLayerIdx(0);
      return;
    }
    const id = setInterval(() => {
      setActiveLayerIdx(i => (i + 1) % ARCH_LAYERS.length);
    }, 220);
    return () => clearInterval(id);
  }, [isRunning]);

  useEffect(() => {
    if (!objPanelOpen) return undefined;

    objPanelFocusRef.current = objPanelTriggerRef.current || document.activeElement;
    objPanelSearchRef.current?.focus();

    const onKey = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setObjPanelOpen(false);
        return;
      }

      if (event.key !== 'Tab' || !objPanelRef.current) return;
      const focusable = objPanelRef.current.querySelectorAll(
        'button, input, select, textarea, [href], [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable.length) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      objPanelFocusRef.current?.focus?.();
      objPanelFocusRef.current = null;
    };
  }, [objPanelOpen]);

  function setTarget(t) {
    const nextTarget = t.trim();
    targetRef.current = nextTarget;
    setTargetState(nextTarget);
    setDraftTarget(nextTarget);
    foundOnceRef.current  = false;
    lastLightRunRef.current = 0;
    lastPredsRef.current = [];
    localTargetRef.current = null;
    trackerRef.current.reset();
    lastAnnouncedSignalRef.current = '';
    lastHeavyRunRef.current = 0;
    aiBoxRef.current = null;
    aiInFlightRef.current = false;
    cloudAiBoxRef.current = null;
    cloudAiInFlightRef.current = false;
    lastCloudAiRunRef.current = 0;
    cloudAiRequestRef.current += 1;
    setServerMs(null);
    setServerLabel('');
    setServerModel('');
    setAlternatives([]);
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
    const direct = resolveCocoTarget(tgt);
    if (direct) {
      localTargetRef.current = { label: direct, score: 1, source: 'alias' };
      return;
    }
    const closest = findClosestCocoLabel(tgt);
    if (closest?.label) {
      localTargetRef.current = { label: closest.label, score: closest.score, source: 'fuzzy' };
    }
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
    cloudAiBoxRef.current = null;
    cloudAiInFlightRef.current = false;
    lastCloudAiRunRef.current = 0;
    cloudAiRequestRef.current += 1;
    setServerMs(null);
    setServerLabel('');
    setServerModel('');
    setAlternatives([]);
  }

  function startScanner(startTarget = targetRef.current) {
    if (isRunningRef.current) return;
    const requestedTarget = typeof startTarget === 'string' ? startTarget.trim() : targetRef.current;
    void sessionRef.current?.start(requestedTarget);
  }

  function stopScanner() {
    sessionRef.current?.stop();
  }

  /** Cancel the startup boot sequence (user tapped Cancel during boot HUD). */
  function cancelBoot() {
    stopScanner();
    // Boot overlay cleanup is handled by the STOP event handler.
  }

  async function toggleTorch() {
    const stream = streamRef.current;
    const nextValue = !torchOn;
    const ok = await setTorch(stream, nextValue);
    if (ok && mountedRef.current && stream === streamRef.current && isRunningRef.current) {
      setTorchOn(nextValue);
    }
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

    const frame = { width: video.videoWidth || 640, height: video.videoHeight || 480 };
    frameRef.current = frame;
    const now = performance.now();
    guidanceTimeRef.current = now;

    const lightInterval = 1000 / lightFpsRef.current;
    const ranLight = now - lastLightRunRef.current >= lightInterval;

    let predictions = lastPredsRef.current;
    if (ranLight) {
      lastLightRunRef.current = now;
      const t0 = performance.now();
      try {
        predictions = await runInference(video);
        if (signal.aborted) return null;
        lastPredsRef.current = Array.isArray(predictions) ? predictions : [];
        predictions = lastPredsRef.current;
        const elapsed = performance.now() - t0;
        recordInferenceTime(elapsed);
        if (mountedRef.current) setCnnMs(Math.round(elapsed));
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn('Inference error', e);
        predictions = lastPredsRef.current;
      }
    }

    if (signal.aborted) return null;

    // ── Heavy path: PulsePointNet server every HEAVY_COOLDOWN_MS ──
    // Runs async and non-blocking; result stored in aiBoxRef for next frame.
    const ranHeavy = now - lastHeavyRunRef.current >= HEAVY_COOLDOWN_MS;
    if (ranHeavy && tgt && !aiInFlightRef.current && isServerAvailable()) {
      lastHeavyRunRef.current = now;
      aiInFlightRef.current = true;
      detectWithServer(video, tgt).then(result => {
        aiInFlightRef.current = false;
        if (signal.aborted || !mountedRef.current) return;
        if (result) {
          aiBoxRef.current = result;
          setServerMs(result.latency_ms ?? null);
          setServerLabel(result.class);
          setServerModel(result.model || '');
          setAlternatives(result.alternatives || []);
        } else {
          aiBoxRef.current = null;
          setServerMs(null);
          setServerLabel('');
          setServerModel('');
          setAlternatives([]);
        }
      });
    }

    const directLabel = resolveCocoTarget(tgt);
    const localInfo = localTargetRef.current;
    const mappedLabel = localInfo?.label || directLabel;
    const priorTrack = trackerRef.current.predict(now);
    const cocoMatchRaw = mappedLabel ? findTarget(predictions, mappedLabel, priorTrack?.bbox, frame) : null;
    const cocoMatch = cocoMatchRaw ? {
      ...cocoMatchRaw,
      displayClass: mappedLabel !== tgt ? tgt : cocoMatchRaw.class,
      source: cocoMatchRaw,
    } : null;

    // ── Cloud AI path: Gemini fallback every CLOUD_AI_COOLDOWN_MS ──
    // Only fires when YOLO and the heavy path have no result. The request is
    // captured from the current camera frame and runs without blocking the
    // on-device detection loop.
    const serverResult = aiBoxRef.current;
    const cloudCooldownOk = now - lastCloudAiRunRef.current >= CLOUD_AI_COOLDOWN_MS;
    if (
      cloudCooldownOk &&
      tgt &&
      !cocoMatch &&
      !serverResult &&
      !aiInFlightRef.current &&
      !cloudAiInFlightRef.current
    ) {
      lastCloudAiRunRef.current = now;
      cloudAiInFlightRef.current = true;
      const requestId = ++cloudAiRequestRef.current;

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = frame.width;
      tempCanvas.height = frame.height;
      const context = tempCanvas.getContext('2d');

      if (context) {
        context.drawImage(video, 0, 0, frame.width, frame.height);
        const base64 = tempCanvas.toDataURL('image/jpeg', 0.75);
        callGeminiBox(base64, tgt).then(result => {
          if (cloudAiRequestRef.current !== requestId) return;
          cloudAiInFlightRef.current = false;
          if (signal.aborted || !mountedRef.current || targetRef.current !== tgt) return;

          if (result?.__error) {
            cloudAiBoxRef.current = null;
            setError('Cloud assist unavailable. Continuing with on-device detection.');
            return;
          }

          if (result?.found) {
            const clamp = value => Math.max(0, Math.min(1, value));
            const x = clamp(result.x);
            const y = clamp(result.y);
            const w = clamp(result.w);
            const h = clamp(result.h);
            cloudAiBoxRef.current = {
              class: tgt,
              score: result.confidence || 0.8,
              bbox: [x * frame.width, y * frame.height, w * frame.width, h * frame.height],
              fromServer: true,
              model: 'Gemini',
              alternatives: [],
              latency_ms: null,
            };
          } else {
            cloudAiBoxRef.current = null;
          }
        }).catch(() => {
          if (cloudAiRequestRef.current !== requestId) return;
          cloudAiInFlightRef.current = false;
          cloudAiBoxRef.current = null;
        });
      } else {
        cloudAiInFlightRef.current = false;
      }
    }

    // ── Merge: use server, then cloud result when YOLO has no match ──
    const freshMatch = cocoMatch || (tgt && serverResult ? {
      ...serverResult,
      displayClass: tgt,
      fromServer: true,
    } : null) || (tgt && cloudAiBoxRef.current ? {
      ...cloudAiBoxRef.current,
      displayClass: tgt,
      fromServer: true,
    } : null);

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
    // ── Boot step: camera ────────────────────────────────────────────────────
    if (mountedRef.current) {
      patchBootStep('camera', { state: 'active', detail: 'Requesting camera permission…' });
    }
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
      if (signal.aborted) return stream;

      if (mountedRef.current) {
        setTorchAvail(hasTorchSupport(stream));
        setTorchOn(false);
        hapticsRef.current.fire('looking', true);
        patchBootStep('camera', { state: 'done', detail: 'Wide-angle feed active.' });
        setBootSteps(prev => {
          const updated = prev.map(s => s.id === 'camera' ? { ...s, state: 'done', detail: 'Wide-angle feed active.' } : s);
          setBootPct(calcBootPct(updated));
          return updated;
        });
      }
      return stream;
    } catch (error) {
      if (mountedRef.current) {
        patchBootStep('camera', { state: 'error', detail: error.message || 'Camera access denied.' });
        setBootActive(false);
      }
      if (stream) stopSessionCamera(stream);
      throw error;
    }
  }

  async function initializeModel({ target: requestedTarget, signal }) {
    // ── Boot step: download + compile + warmup ───────────────────────────────
    const onProgress = (progress) => {
      if (!mountedRef.current || signal.aborted) return;

      if (progress.step === 'download') {
        setBootSteps(prev => {
          const updated = prev.map(s => {
            if (s.id !== 'download') return s;
            const detail = progress.fromCache
              ? 'Loaded from offline cache.'
              : progress.total > 0
                ? `${(progress.loaded / 1_048_576).toFixed(1)} / ${(progress.total / 1_048_576).toFixed(1)} MB (${progress.percent}%)`
                : `${(progress.loaded / 1_048_576).toFixed(1)} MB…`;
            const isDone = progress.percent >= 100;
            return { ...s, state: isDone ? 'done' : 'active', subPercent: progress.percent, detail };
          });
          setBootPct(calcBootPct(updated));
          return updated;
        });
        // detect first-boot from non-cache download
        if (!progress.fromCache && progress.percent === 0 && mountedRef.current) {
          setIsFirstBoot(true);
        }
      }

      if (progress.step === 'compile') {
        setBootSteps(prev => {
          const isDone = progress.percent >= 100;
          const updated = prev.map(s => {
            if (s.id === 'download') return { ...s, state: 'done' };
            if (s.id === 'compile') return { ...s, state: isDone ? 'done' : 'active', detail: progress.message || 'Compiling WASM SIMD graph…', subPercent: null };
            return s;
          });
          setBootPct(calcBootPct(updated));
          return updated;
        });
      }

      if (progress.step === 'warmup') {
        setBootSteps(prev => {
          const isDone = progress.percent >= 100;
          const updated = prev.map(s => {
            if (s.id === 'compile') return { ...s, state: 'done' };
            if (s.id === 'warmup') return { ...s, state: isDone ? 'done' : 'active', detail: progress.message || 'Pre-compiling JIT kernels…', subPercent: null };
            return s;
          });
          setBootPct(calcBootPct(updated));
          return updated;
        });
      }
    };

    // Activate the download step before we start (compile and warmup start later).
    if (mountedRef.current) {
      patchBootStep('download', { state: 'active', detail: 'Preparing neural weights…', subPercent: 0 });
    }

    const model = await loadModel({ onProgress, signal });

    if (!signal.aborted && mountedRef.current) {
      // All 4 steps done — close the boot overlay and hand off to active scanning.
      setBootSteps(makeSteps); // reset for next session
      setBootPct(100);
      // Brief display of 100% before dismounting
      setTimeout(() => {
        if (mountedRef.current) {
          setBootActive(false);
          setBootPct(0);
          setStatus('looking');
          setAnnouncement(requestedTarget ? `Looking for ${requestedTarget}.` : 'Camera active. Say or type a target.');
          speakerRef.current.say('Pulse Point ready.', { urgent: true });
        }
      }, 350);
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
      setTorchOn(false);
      setTorchAvail(false);
      // Close the boot overlay if the user cancels during startup
      setBootActive(false);
      setBootSteps(makeSteps);
      setBootPct(0);
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
        // Reset steps and show boot HUD
        setBootSteps(makeSteps);
        setBootPct(0);
        setIsFirstBoot(!isModelReady());
        setBootActive(true);
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
        setBootActive(false);
        setBootSteps(makeSteps);
        setBootPct(0);
        setError(event.error?.message || 'Camera blocked. Allow camera access and try again.');
        setTorchOn(false);
        setTorchAvail(false);
        break;
      case SCANNER_EVENTS.MODEL_ERROR:
        isRunningRef.current = false;
        setIsRunning(false);
        setStatus('blocked');
        setBootActive(false);
        setBootSteps(makeSteps);
        setBootPct(0);
        setError(event.error?.message || 'CNN model failed to load. Refresh and try again.');
        setTorchOn(false);
        setTorchAvail(false);
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
    setCnnConf(Math.round(displayMatch.score * 100));
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
    canvas.width  = displayW;
    canvas.height = displayH;
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

  // Build alternatives list for confidence histogram
  const confBars = (() => {
    if (match && cnnConf != null) {
      const top = [{ name: match.name, confidence: cnnConf / 100 }];
      const alts = alternatives.slice(0, 4).filter(a => a.name !== match.name);
      return [...top, ...alts].slice(0, 5);
    }
    return alternatives.slice(0, 5);
  })();

  return (
    <>
      <a href="#target-input" className="sr-only skip-link">Skip to search</a>
      {showWelcome && <WelcomeOverlay onDismiss={dismissWelcome} />}
      {bootActive && (
        <StartupProgress
          steps={bootSteps}
          overallPct={bootPct}
          onCancel={cancelBoot}
          isFirstBoot={isFirstBoot}
        />
      )}

      <main className={`scanner signal-${signal}`}>
      <h1 className="sr-only">Pulse Point Object Finder</h1>
      <video ref={videoRef} playsInline muted aria-hidden="true" />
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={match ? `Detection overlay: ${match.name} ${match.direction}, ${match.distance}` : 'Detection overlay'}
      />

      <Announcer message={announcement} urgent={announcementUrgent} />

      {!isRunning && (
        <button className="start-button" type="button" onClick={startScanner} aria-label="Start CNN scanner">
          <Camera size={30} aria-hidden="true" />
          <span>Start</span>
        </button>
      )}

      {/* CNN status badge — top-left */}
      <div className={`cnn-badge${isRunning ? ' cnn-active' : ''}`} aria-hidden="true">
        <span className="cnn-dot" />
        <span className="cnn-label">CNN</span>
        {isRunning && cnnMs != null && <span className="cnn-ms">{cnnMs}ms</span>}
      </div>

      {/* Architecture ribbon — top-center, only when running */}
      {isRunning && (
        <div className="arch-ribbon" aria-hidden="true">
          {ARCH_LAYERS.map((layer, i) => (
            <React.Fragment key={layer.id}>
              <div className={`arch-node${i === activeLayerIdx ? ' arch-active' : i < activeLayerIdx ? ' arch-done' : ''}`}>
                <span className="arch-node-name">{layer.label}</span>
                <span className="arch-node-dim">{layer.dim}</span>
              </div>
              {i < ARCH_LAYERS.length - 1 && <span className="arch-arrow">›</span>}
            </React.Fragment>
          ))}
        </div>
      )}

      {/* Top-right control rail */}
      <div className="top-rail" role="group" aria-label="Quick controls">
        {torchAvail && (
          <button
            type="button"
            className={`rail-btn${torchOn ? ' active' : ''}`}
            onClick={toggleTorch}
            aria-label={torchOn ? 'Turn flashlight off' : 'Turn flashlight on'}
            aria-pressed={torchOn}
          >
            {torchOn ? <Flashlight size={18} aria-hidden="true" /> : <FlashlightOff size={18} aria-hidden="true" />}
          </button>
        )}
        <button
          type="button"
          className="rail-btn"
          ref={objPanelTriggerRef}
          onClick={() => setObjPanelOpen(v => !v)}
          aria-label="Show trained objects"
          aria-expanded={objPanelOpen}
          aria-controls="trained-objects-panel"
          title="Trained objects"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        </button>
        <button
          type="button"
          className="rail-btn"
          onClick={() => setSettingsOpen(true)}
          aria-label="Open settings"
        >
          <SettingsIcon size={18} aria-hidden="true" />
        </button>
      </div>

      {/* Feature activation grid — bottom-left */}
      {isRunning && (
        <FeatureGrid active={!!match} confidence={cnnConf ?? 0} />
      )}

      {/* Layer depth panel — right side */}
      {isRunning && (
        <div className="layer-panel" aria-hidden="true">
          {LAYER_STACK.map((layer, i) => {
            const isActive = match && i === Math.floor(activeLayerIdx / ARCH_LAYERS.length * LAYER_STACK.length);
            return (
              <div key={i} className={`layer-row${isActive ? ' layer-active' : ''}`}>
                <div className="layer-bar-track">
                  <div
                    className="layer-bar-fill"
                    style={{ height: `${(match ? layer.fill : layer.fill * 0.25) * 100}%` }}
                  />
                </div>
                <div className="layer-info">
                  <span className="layer-name">{layer.name}</span>
                  <span className="layer-dim">{layer.dim}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* CNN inference stats bar */}
      {isRunning && (
        <div className="cnn-stats" aria-hidden="true">
          <span className="cnn-stat-item">
            <span className="cnn-stat-label">INFER</span>
            <span className="cnn-stat-val">{cnnMs != null ? `${cnnMs} ms` : '—'}</span>
          </span>
          <span className="cnn-stat-sep" />
          <span className="cnn-stat-item">
            <span className="cnn-stat-label">CONF</span>
            <span className="cnn-stat-val">{cnnConf != null ? `${cnnConf}%` : '—'}</span>
          </span>
          <span className="cnn-stat-sep" />
          <span className="cnn-stat-item">
            <span className="cnn-stat-label">ANCHORS</span>
            <span className="cnn-stat-val">8400</span>
          </span>
          <span className="cnn-stat-sep" />
          <span className={`cnn-stat-item${serverMs != null || cloudAiBoxRef.current ? ' cnn-server-active' : ''}`}>
            <span className="cnn-stat-label">GRND</span>
            <span className="cnn-stat-val">
              {serverMs != null
                ? `${serverMs} ms`
                : cloudAiBoxRef.current ? 'cloud'
                : isServerAvailable() ? 'ready' : 'off'}
            </span>
          </span>
        </div>
      )}

      {/* Confidence histogram — bottom-right */}
      {isRunning && confBars.length > 0 && (
        <div className="conf-hist" aria-hidden="true">
          <div className="conf-hist-title">CLASS PROB</div>
          {confBars.map((bar, i) => (
            <div key={i} className="conf-bar-row">
              <span className="conf-bar-label">{bar.name}</span>
              <div className="conf-bar-track">
                <div
                  className={`conf-bar-fill ${i === 0 ? 'top' : 'alt'}`}
                  style={{ width: `${Math.round((bar.confidence ?? 0) * 100)}%` }}
                />
              </div>
              <span className="conf-bar-pct">{Math.round((bar.confidence ?? 0) * 100)}%</span>
            </div>
          ))}
        </div>
      )}

      {/* Trained objects drawer */}
      {objPanelOpen && (
        <div
          id="trained-objects-panel"
          className="obj-panel"
          role="dialog"
          aria-modal="true"
          aria-labelledby="trained-objects-title"
          ref={objPanelRef}
        >
          <div className="obj-panel-header">
            <span id="trained-objects-title" className="obj-panel-title">Trained Objects <span className="obj-panel-count">{KNOWN_OBJECTS.length}</span></span>
            <button type="button" className="obj-panel-close" onClick={() => setObjPanelOpen(false)} aria-label="Close trained objects">×</button>
          </div>
          <input
            ref={objPanelSearchRef}
            className="obj-panel-search"
            type="text"
            placeholder="filter…"
            value={objFilter}
            onChange={e => setObjFilter(e.target.value)}
            autoComplete="off"
          />
          <div className="obj-panel-list">
            {(objFilter
              ? KNOWN_OBJECTS.filter(o => o.includes(objFilter.toLowerCase()))
              : KNOWN_OBJECTS
            ).map(obj => (
              <button
                key={obj}
                type="button"
                className="obj-chip"
                onClick={() => {
                  setObjPanelOpen(false);
                  setError('');
                  setTarget(obj);
                  setDraftTarget(obj);
                  if (!isRunningRef.current) startScanner(obj);
                  else setStatus('looking');
                }}
              >
                {obj}
              </button>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="error-pill" role="alert">
          {error}
          <button type="button" className="error-dismiss" onClick={() => setError('')} aria-label="Dismiss error">×</button>
        </div>
      )}

      <div className="scan-sweep" aria-hidden="true" />

      <div className="reticle" aria-hidden="true">
        <div className="reticle-inner">
          <div className="reticle-corner reticle-corner--tl" />
          <div className="reticle-corner reticle-corner--tr" />
          <div className="reticle-corner reticle-corner--bl" />
          <div className="reticle-corner reticle-corner--br" />
          <div className="reticle-dot" />
          <div className="reticle-scan" />
        </div>
      </div>

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

      <div className="signal-strip" aria-live="polite">
        <div className="signal-strip-dot" aria-hidden="true" />
        <div className="signal-strip-text">
          <strong>
            {status === 'idle'    ? 'idle'    :
             status === 'booting' ? 'booting' :
             status}
          </strong>
          <span>
            {match
              ? `${match.direction} · ${match.distance}`
              : status === 'idle'    ? 'tap start to scan'
              : status === 'booting' ? 'starting engine…'
              : isRunning ? 'inference running…'
              : 'point camera at object'}
          </span>
        </div>
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
  const norm = normalizeTargetText(target);
  const aliases = TARGET_ALIASES[norm] ? [TARGET_ALIASES[norm]] : [norm];

  return predictions
    .filter(p => aliases.includes(p.class.toLowerCase()))
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
