import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Camera, Loader2, ScanLine, Square, Mic, Settings as SettingsIcon, Flashlight, FlashlightOff } from 'lucide-react';

import { resolveCocoTarget, findClosestCocoLabel, TARGET_ALIASES, normalizeTargetText } from './detection/coco.js';
import { loadModel, runInference } from './detection/engine.js';
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

const ADAPTIVE_FPS_MIN = 4;
const ADAPTIVE_FPS_MAX = 15;
const ADAPTIVE_FPS_INITIAL = 10;
const ADAPTIVE_SLACK_MS = 25;
const HEAVY_COOLDOWN_MS = 2500;

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
  const [status,        setStatus]        = useState('ready');
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
  const [cnnMs,         setCnnMs]         = useState(null);   // last CNN inference latency
  const [cnnConf,       setCnnConf]       = useState(null);   // last detection confidence
  const [serverMs,      setServerMs]      = useState(null);   // last server inference latency
  const [serverLabel,   setServerLabel]   = useState('');     // last server detected label
  const [serverModel,   setServerModel]   = useState('');     // 'LocateAnything-3B' or 'PulsePointNet'
  const [objPanelOpen,  setObjPanelOpen]  = useState(false);  // trained-objects drawer
  const [objFilter,     setObjFilter]     = useState('');

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

  useEffect(() => {
    hapticsRef.current.setEnabled(settings.haptics);
    speakerRef.current.setEnabled(settings.speech);
    speakerRef.current.setRate(settings.speechRate);
    saveSettings(settings);
  }, [settings]);

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
    setServerMs(null);
    setServerLabel('');
    setServerModel('');
    resolveLocalTarget(nextTarget);
    sessionRef.current?.setTarget(nextTarget);
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
    setServerMs(null);
    setServerLabel('');
    setServerModel('');
  }

  function startScanner(startTarget = targetRef.current) {
    if (isRunningRef.current) return;
    const requestedTarget = typeof startTarget === 'string' ? startTarget.trim() : targetRef.current;
    void sessionRef.current?.start(requestedTarget);
  }

  function stopScanner() {
    sessionRef.current?.stop();
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
        } else {
          aiBoxRef.current = null;
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

    // ── Merge: use server result when YOLO has no match for the target ──
    const serverResult = aiBoxRef.current;
    const freshMatch = cocoMatch || (tgt && serverResult ? {
      ...serverResult,
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
        setStatus('loading');
        setAnnouncement('Initializing CNN model weights…');
        setAnnouncementUrgent(false);
        hapticsRef.current.fire('looking', true);
      }
      return stream;
    } catch (error) {
      if (stream) stopSessionCamera(stream);
      throw error;
    }
  }

  async function initializeModel({ target: requestedTarget, signal }) {
    const model = await loadModel();
    if (!signal.aborted && mountedRef.current) {
      setStatus('looking');
      setAnnouncement(requestedTarget ? `Looking for ${requestedTarget}.` : 'Camera active. Say or type a target.');
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
      setStatus('ready');
      setMatch(null);
      setSignal('looking');
      resetDetectionState();
      setTorchOn(false);
      setTorchAvail(false);
      return;
    }

    if (!mountedRef.current) return;

    switch (event.type) {
      case SCANNER_EVENTS.START:
        isRunningRef.current = true;
        setError('');
        setIsRunning(true);
        setStatus('camera');
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
        setTorchOn(false);
        setTorchAvail(false);
        break;
      case SCANNER_EVENTS.MODEL_ERROR:
        isRunningRef.current = false;
        setIsRunning(false);
        setStatus('blocked');
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
    const displayW = canvas.clientWidth || W;
    const displayH = canvas.clientHeight || H;
    canvas.width = displayW;
    canvas.height = displayH;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, displayW, displayH);

    const scale = Math.max(displayW / W, displayH / H);
    const offsetX = (W * scale - displayW) / 2;
    const offsetY = (H * scale - displayH) / 2;
    const mapBox = ([x, y, w, h]) => [x * scale - offsetX, y * scale - offsetY, w * scale, h * scale];

    if (settingsRef.current.showAllBoxes) {
      predictions.forEach(p => {
        if (targetMatch?.source === p) return;
        const [x, y, w, h] = mapBox(p.bbox);
        ctx.strokeStyle = 'rgba(255,255,255,0.28)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.strokeRect(x, y, w, h);
      });
    }

    if (!targetMatch) return;

    const [x, y, bw, bh] = mapBox(targetMatch.bbox);
    // visualize extrapolation: fade box as ageMs grows
    const opacity = targetMatch.ageMs > 100 ? 0.7 : 1;
    ctx.globalAlpha = opacity;
    ctx.strokeStyle = '#00ff9d';
    ctx.lineWidth   = 8;
    ctx.strokeRect(x, y, bw, bh);
    ctx.globalAlpha = 1;

    const display = targetMatch.displayClass || targetMatch.class;
    const label = `${display} ${Math.round(targetMatch.score * 100)}%`;
    const labelW = Math.min(280, bw);
    ctx.fillStyle = '#00ff9d';
    ctx.fillRect(x, Math.max(0, y - 38), labelW, 38);
    ctx.fillStyle = '#05100d';
    ctx.font = '800 20px system-ui';
    ctx.fillText(label, x + 10, Math.max(26, y - 12));
  }

  return (
    <main className={`scanner signal-${signal}`}>
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
        {isRunning && cnnMs != null && (
          <span className="cnn-ms">{cnnMs}ms</span>
        )}
      </div>

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
          onClick={() => setObjPanelOpen(v => !v)}
          aria-label="Show trained objects"
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

      {status === 'loading' && (
        <div className="loading-pill" role="status">
          <Loader2 size={18} aria-hidden="true" />
          <span>Loading CNN weights…</span>
        </div>
      )}

      {/* CNN inference stats bar — shows when running */}
      {isRunning && (
        <div className="cnn-stats" aria-hidden="true">
          <span className="cnn-stat-item">
            <span className="cnn-stat-label">YOLO</span>
            <span className="cnn-stat-val">{cnnMs != null ? `${cnnMs} ms` : '—'}</span>
          </span>
          <span className="cnn-stat-sep" />
          <span className="cnn-stat-item">
            <span className="cnn-stat-label">CONF</span>
            <span className="cnn-stat-val">{cnnConf != null ? `${cnnConf}%` : '—'}</span>
          </span>
          <span className="cnn-stat-sep" />
          <span className={`cnn-stat-item${serverMs != null ? ' cnn-server-active' : ''}`}>
            <span className="cnn-stat-label">{serverModel === 'LocateAnything-3B' ? 'LA-3B' : 'PPN'}</span>
            <span className="cnn-stat-val">
              {serverMs != null
                ? `${serverMs} ms${serverLabel ? ` · ${serverLabel}` : ''}`
                : isServerAvailable() ? 'ready' : 'offline'}
            </span>
          </span>
        </div>
      )}

      {/* Trained objects drawer */}
      {objPanelOpen && (
        <div className="obj-panel" role="dialog" aria-label="Trained object classes">
          <div className="obj-panel-header">
            <span className="obj-panel-title">Trained Objects <span className="obj-panel-count">{KNOWN_OBJECTS.length}</span></span>
            <button type="button" className="obj-panel-close" onClick={() => setObjPanelOpen(false)} aria-label="Close">×</button>
          </div>
          <input
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
          <button
            type="button"
            className="error-dismiss"
            onClick={() => setError('')}
            aria-label="Dismiss error"
          >
            ×
          </button>
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
          <strong>{status}</strong>
          <span>
            {match
              ? `${match.direction} · ${match.distance}`
              : isRunning ? 'CNN scanning…' : 'point camera at object'}
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
