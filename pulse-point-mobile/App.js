import React, { useEffect, useRef, useState, useReducer } from 'react';
import {
  Animated,
  Easing,
  Platform,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Magnetometer } from 'expo-sensors';
import * as Speech from 'expo-speech';
import { StatusBar } from 'expo-status-bar';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

import {
  detectObject,
  checkHealth,
  apiResultToDetection,
  isSimulationOptedIn,
} from './src/services/visionAPI';
import { CaptureQueue } from './src/services/captureQueue';
import {
  createInitialGuidanceState,
  EVENTS,
  mobileGuidanceReducer,
} from './src/services/guidanceState';
import {
  DirectionHapticEngine,
  playTransitionHaptic,
  directionArrow,
  directionLabel,
  Direction,
  Proximity,
} from './src/services/directionHaptics';

// ── Detection config ──────────────────────────────────────────────────────────

// ── Error boundary ────────────────────────────────────────────────────────────

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidCatch(err, info) { console.error('Pulse Point crashed', err, info); }
  handleRetry = () => this.setState({ hasError: false });

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <SafeAreaView style={styles.permissionScreen}>
        <StatusBar style="light" />
        <Text style={styles.permissionTitle}>Something went wrong</Text>
        <Text style={styles.permissionText}>Try again. If it keeps failing, restart the app.</Text>
        <Pressable style={styles.primaryButton} onPress={this.handleRetry}>
          <Text style={styles.primaryButtonText}>Try again</Text>
        </Pressable>
      </SafeAreaView>
    );
  }
}

// ── Main component ────────────────────────────────────────────────────────────

function AppContent() {
  const [permission, requestPermission] = useCameraPermissions();

  // Core state
  const [target, setTarget]           = useState('');
  const [guidanceState, dispatch] = useReducer(
    mobileGuidanceReducer,
    undefined,
    createInitialGuidanceState,
  );
  const [visionMode, setVisionMode]   = useState('checking');
  const [micPrimed, setMicPrimed]     = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [healthAttempt, setHealthAttempt] = useState(0);

  // Direction overlay state (driven by haptic engine callbacks)
  const [direction, setDirection]     = useState(null);
  const [proximity, setProximity]     = useState(null);
  const [distanceM, setDistanceM]     = useState(null);
  const [healthError, setHealthError] = useState(null);

  // Sensor
  const [heading, setHeading]         = useState(0);

  // Animations
  const scanPulse   = useRef(new Animated.Value(0)).current;
  const proxRing    = useRef(new Animated.Value(0)).current;
  const dirFade     = useRef(new Animated.Value(0)).current;

  // Refs
  const cameraRef           = useRef(null);
  const hapticEngineRef     = useRef(null);
  const captureQueueRef     = useRef(null);
  const guidanceStateRef    = useRef(guidanceState);
  const previousTargetRef   = useRef(target);
  const mountedRef          = useRef(true);
  const hadDetectionRef     = useRef(false);
  const inputRef            = useRef(null);
  const speechThrottleRef   = useRef(null);     // last direction spoken
  const simulationOptedIn   = isSimulationOptedIn();

  guidanceStateRef.current = guidanceState;
  const stageKey = guidanceState.status;
  const detection = guidanceState.detection;
  const stage = stageForGuidance(guidanceState, target);
  const isRunning = !['idle', 'error'].includes(guidanceState.status);
  const canScan   = Boolean(target.trim());

  function stopSession() {
    captureQueueRef.current?.stop();
    captureQueueRef.current = null;
    hapticEngineRef.current?.stop();
    hadDetectionRef.current = false;
    if (mountedRef.current) {
      setDirection(null);
      setProximity(null);
      setDistanceM(null);
    }
  }

  // ── Engine init ────────────────────────────────────────────────────────────

  useEffect(() => {
    hapticEngineRef.current = new DirectionHapticEngine();
    hapticEngineRef.current.onUpdate((dir, prox, dist) => {
      if (!mountedRef.current) return;
      setDirection(dir);
      setProximity(prox);
      setDistanceM(dist);
      _animateProxRing(prox);
      _throttledDirectionSpeech(dir, prox);
    });

    return () => hapticEngineRef.current?.stop();
  }, []);

  // ── Server health check ────────────────────────────────────────────────────

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setVisionMode('checking');
    checkHealth({ signal: controller.signal }).then(result => {
      if (!active || !mountedRef.current) return;
      setVisionMode(result.ok ? 'server' : 'unavailable');
      setHealthError(result.ok ? null : result.error?.message || 'Vision API is unavailable.');
    }).catch(error => {
      if (!active || !mountedRef.current || error?.name === 'AbortError') return;
      setVisionMode('unavailable');
      setHealthError(error.message || 'Vision API is unavailable.');
    });
    return () => {
      active = false;
      controller.abort();
    };
  }, [healthAttempt]);

  // ── Compass ───────────────────────────────────────────────────────────────

  useEffect(() => {
    Magnetometer.setUpdateInterval(300);
    const sub = Magnetometer.addListener(({ x, y }) => {
      let angle = Math.atan2(y, x) * (180 / Math.PI);
      angle = (angle + 360) % 360;
      const h = Platform.OS === 'android' ? (360 - angle + 90) % 360 : angle;
      if (mountedRef.current) setHeading(Math.round(h));
    });
    return () => sub.remove();
  }, []);

  // ── Scan-sweep animation ───────────────────────────────────────────────────

  useEffect(() => {
    const anim = Animated.loop(
      Animated.timing(scanPulse, {
        toValue: 1,
        duration: ['loading', 'looking', 'reacquiring'].includes(stageKey) ? 850 : 1300,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      })
    );
    anim.start();
    return () => anim.stop();
  }, [scanPulse, stageKey]);

  // ── Direction arrow fade ───────────────────────────────────────────────────

  useEffect(() => {
    if (direction) {
      Animated.timing(dirFade, { toValue: 1, duration: 180, useNativeDriver: true }).start();
    } else {
      Animated.timing(dirFade, { toValue: 0, duration: 300, useNativeDriver: true }).start();
    }
  }, [direction, dirFade]);

  // ── Speech for stage transitions ───────────────────────────────────────────

  useEffect(() => {
    if (stageKey === 'idle') {
      Speech.stop();
      return;
    }
    const txt = _buildSpeech(stageKey, stage, target);
    Speech.stop();
    Speech.speak(txt, {
      language: 'en-US',
      pitch: 1,
      rate: Platform.OS === 'ios' ? 0.48 : 0.86,
    });
  }, [stageKey]);

  // ── Cleanup, target changes, and permission loss ───────────────────────────

  useEffect(() => {
    return () => {
      mountedRef.current = false;
      stopSession();
      Speech.stop();
    };
  }, []);

  useEffect(() => {
    const previous = previousTargetRef.current;
    previousTargetRef.current = target;
    if (previous === target) return;
    Speech.stop();
    speechThrottleRef.current = null;
    if (captureQueueRef.current?.isRunning() || guidanceStateRef.current.status !== 'idle') {
      stopSession();
      dispatch({ type: EVENTS.RESET });
    }
    dispatch({ type: EVENTS.TARGET_SET, target });
  }, [target]);

  useEffect(() => {
    if (!permission || permission.granted) return;
    stopSession();
    setCameraActive(false);
    dispatch({ type: EVENTS.CAMERA_ERROR, error: 'Camera permission is required to scan. Allow camera access, then try again.' });
  }, [permission?.granted]);

  // ── Single-flight capture and request loop ─────────────────────────────────

  function handleDetectionResult(apiResult, targetName, queue) {
    if (!mountedRef.current || captureQueueRef.current !== queue) return;
    const nowMs = Date.now();
    const detectionResult = apiResultToDetection(apiResult, targetName, nowMs);
    const event = detectionResult
      ? { type: EVENTS.DETECTION_RECEIVED, detection: detectionResult, nowMs }
      : { type: EVENTS.DETECTION_MISSED, nowMs };
    const nextState = mobileGuidanceReducer(guidanceStateRef.current, event);
    dispatch(event);

    if (detectionResult) {
      if (!hadDetectionRef.current) {
        hadDetectionRef.current = true;
        playTransitionHaptic('found');
      }
      _feedEngine(detectionResult, nextState);
    } else if (nextState.status === 'lost') {
      hapticEngineRef.current?.stop();
      playTransitionHaptic('lost');
    }
  }

  function beginQueue(targetName, mode = 'server') {
    let simulationFrame = 0;
    const queue = new CaptureQueue({
      intervalMs: 800,
      requestTimeoutMs: 5500,
      maxRetries: 2,
      capture: async ({ signal }) => {
        if (mode === 'simulation') return { simulated: true };
        if (!cameraRef.current) return null;
        return cameraRef.current.takePictureAsync({
          quality: 0.45,
          base64: false,
          skipProcessing: true,
          signal,
        });
      },
      request: (frame, { target: requestedTarget, signal, timeoutMs }) => {
        if (mode === 'simulation') {
          const phase = simulationFrame++ % 4;
          return Promise.resolve({
            detected: true,
            name: requestedTarget,
            confidence: 0.9,
            boundingBox: { x: [0.18, 0.32, 0.46, 0.58][phase], y: 0.45, width: 0.08, height: 0.08 },
            distance: { meters: 3, method: 'area-estimate', uncertaintyMeters: 2 },
            source: 'simulated',
          });
        }
        return detectObject(frame.uri, requestedTarget, { signal, timeoutMs });
      },
      onResult: ({ result }) => handleDetectionResult(result, targetName, queue),
      onMiss: () => handleDetectionResult(null, targetName, queue),
      onError: ({ error }) => {
        if (!mountedRef.current || captureQueueRef.current !== queue) return;
        captureQueueRef.current?.stop();
        captureQueueRef.current = null;
        setVisionMode('unavailable');
        setHealthError(error?.message || 'The vision service stopped responding. Check the server and retry.');
        dispatch({ type: EVENTS.MODEL_ERROR, error: 'Vision service unavailable. Check the server connection, then retry.' });
      },
    });
    captureQueueRef.current = queue;
    queue.start(targetName);
  }

  function startScan() {
    const targetName = target.trim();
    if (!targetName) { setCameraActive(true); inputRef.current?.focus(); return; }
    if (!permission?.granted) {
      dispatch({ type: EVENTS.CAMERA_ERROR, error: 'Camera permission is required to scan.' });
      return;
    }
    if (visionMode === 'checking') {
      dispatch({ type: EVENTS.MODEL_ERROR, error: 'Still checking the vision service. Try again in a moment.' });
      return;
    }
    if (visionMode !== 'server') {
      dispatch({ type: EVENTS.MODEL_ERROR, error: 'Vision is unavailable. Start the configured server or use the explicit demo button.' });
      return;
    }
    stopSession();
    setCameraActive(true);
    dispatch({ type: EVENTS.START, target: targetName });
    beginQueue(targetName);
  }

  function startSimulation() {
    const targetName = target.trim();
    if (!simulationOptedIn || !targetName) return;
    stopSession();
    setVisionMode('simulation');
    setCameraActive(true);
    dispatch({ type: EVENTS.START, target: targetName });
    beginQueue(targetName, 'simulation');
  }

  function resetScan() {
    stopSession();
    dispatch({ type: EVENTS.RESET });
    setCameraActive(false);
    Speech.stop();
  }

  function _feedEngine(det, state) {
    if (!det?.bbox) return;
    hapticEngineRef.current?.update(det.bbox, {
      allowReach: Boolean(state?.assistiveReady && state.status === 'reach'),
    });
  }

  function _animateProxRing(prox) {
    const speeds = {
      [Proximity.FAR]:    1400,
      [Proximity.MEDIUM]:  900,
      [Proximity.CLOSE]:   520,
      [Proximity.NEAR]:    260,
      [Proximity.REACH]:    90,
    };
    const dur = speeds[prox] ?? 1000;
    Animated.loop(
      Animated.sequence([
        Animated.timing(proxRing, { toValue: 1, duration: dur * 0.45, easing: Easing.out(Easing.ease), useNativeDriver: true }),
        Animated.timing(proxRing, { toValue: 0, duration: dur * 0.55, easing: Easing.in(Easing.ease),  useNativeDriver: true }),
      ]),
      { iterations: 1 }
    ).start();
  }

  // Speak direction changes at most every 3 seconds — don't spam voice
  function _throttledDirectionSpeech(dir, prox) {
    if (prox === Proximity.REACH) return;

    const now = Date.now();
    if (speechThrottleRef.current && now - speechThrottleRef.current < 3000) return;
    speechThrottleRef.current = now;

    const phrases = {
      [Direction.LEFT]:   'move left',
      [Direction.RIGHT]:  'move right',
      [Direction.UP]:     'aim up',
      [Direction.DOWN]:   'aim down',
      [Direction.LOCKED]: null,
    };
    const phrase = phrases[dir];
    if (phrase) {
      Speech.speak(phrase, { language: 'en-US', rate: 0.55 });
    }
  }

  // ── Proximity ring styles ──────────────────────────────────────────────────

  const proxRingStyle = () => {
    const colors = {
      [Proximity.FAR]:    '#118ab2',
      [Proximity.MEDIUM]: '#06d6a0',
      [Proximity.CLOSE]:  '#ffd166',
      [Proximity.NEAR]:   '#ef476f',
      [Proximity.REACH]:  '#00ff9d',
    };
    const color = proximity ? colors[proximity] : 'rgba(255,255,255,0.3)';
    return {
      borderColor: color,
      opacity: proxRing.interpolate({ inputRange: [0, 1], outputRange: [0.3, 0.95] }),
      transform: [{
        scale: proxRing.interpolate({ inputRange: [0, 1], outputRange: [1.0, 1.18] }),
      }],
    };
  };

  // ── Direction arrow colour ─────────────────────────────────────────────────

  const dirArrowColor = () => {
    if (!direction) return '#ffffff';
    if (direction === Direction.REACH)  return '#00ff9d';
    if (direction === Direction.LOCKED) return '#06d6a0';
    return '#ffffff';
  };

  // ── Permission screens ─────────────────────────────────────────────────────

  if (!permission) {
    return (
      <View style={styles.permissionScreen}>
        <StatusBar style="light" />
        <Text style={styles.permissionTitle}>Loading Pulse Point</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permissionScreen}>
        <StatusBar style="light" />
        <MaterialCommunityIcons name="camera-iris" size={58} color="#06d6a0" />
        <Text style={styles.permissionTitle}>Pulse Point needs camera access</Text>
        <Text style={styles.permissionText}>
          The app scans your room to locate the object you're looking for and guides you to it with haptic feedback.
        </Text>
        <Pressable style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryButtonText}>Allow camera</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  // ── Main render ────────────────────────────────────────────────────────────

  const handleScanButton = () => {
    if (isRunning) { resetScan(); return; }
    if (canScan)   { startScan(); return; }
    setCameraActive(true);
  };

  const scanSweepTranslate = scanPulse.interpolate({
    inputRange: [0, 1], outputRange: [-80, 760],
  });
  const modeBadgeStyle = visionMode === 'server'
    ? styles.modeBadgeServer
    : visionMode === 'simulation'
      ? styles.modeBadgeSimulation
      : styles.modeBadgeUnavailable;
  const modeLabel = {
    checking: 'CHECKING',
    server: 'SERVER',
    unavailable: 'UNAVAILABLE',
    simulation: 'DEMO ONLY',
  }[visionMode] || 'UNAVAILABLE';

  return (
    <View style={styles.app}>
      <StatusBar style="light" />

      {/* Camera */}
      <CameraView ref={cameraRef} style={styles.camera} facing="back" autofocus="on" />
      <View style={styles.cameraShade} />

      {/* Scan sweep line */}
      <Animated.View
        pointerEvents="none"
        style={[styles.scanSweep, {
          opacity: (cameraActive || isRunning) ? 1 : 0.4,
          transform: [{ translateY: scanSweepTranslate }],
        }]}
      />

      <SafeAreaView style={styles.overlay}>
        {/* ── Status strip ─────────────────────────────────────────── */}
        <View style={styles.signalStrip}>
          <View style={styles.signalHeader}>
            <Text style={styles.signalStatus}>{stage.label}</Text>
            <View style={[styles.modeBadge, modeBadgeStyle]}>
              <Text style={styles.modeBadgeText}>{modeLabel}</Text>
            </View>
          </View>

          <Text style={styles.signalDetail}>
            {detection
              ? `${stage.instruction} · ${distanceM != null ? `${distanceM.toFixed(1)} m` : guidanceState.distanceText || 'distance unconfirmed'}`
              : isRunning ? stage.instruction
              : cameraActive ? 'camera live — type a target'
              : 'ready'}
          </Text>

          {visionMode === 'server' && (
            <Text style={styles.healthText}>Server detector is experimental; reach is disabled.</Text>
          )}
          {visionMode === 'unavailable' && !isRunning && (
            <View style={styles.healthActions}>
              <Text style={styles.healthText}>{healthError || 'Configure and start the vision server, then retry.'}</Text>
              <Pressable style={styles.retryButton} onPress={() => setHealthAttempt(value => value + 1)}>
                <Text style={styles.retryButtonText}>Retry connection</Text>
              </Pressable>
              {simulationOptedIn && canScan && (
                <Pressable style={styles.demoButton} onPress={startSimulation}>
                  <Text style={styles.retryButtonText}>Run explicit demo (not assistive)</Text>
                </Pressable>
              )}
            </View>
          )}
          {visionMode === 'simulation' && (
            <Text style={styles.healthText}>Demo only. No assistive readiness or reach signal.</Text>
          )}
          {detection?.apiMeta?.latencyMs != null && (
            <Text style={styles.latencyText}>{detection.apiMeta.latencyMs} ms · {Math.round((detection.confidence ?? 0) * 100)}% conf</Text>
          )}
        </View>

        {/* ── Heading + direction badge (top-right) ─────────────────── */}
        <View style={styles.topRail}>
          <View style={styles.sensorPill}>
            <Ionicons name="navigate" size={15} color="#06d6a0" />
            <Text style={styles.sensorPillText}>{heading}°</Text>
          </View>

          {direction && (
            <Animated.View style={[styles.dirBadge, { opacity: dirFade }]}>
              <Text style={[styles.dirArrow, { color: dirArrowColor() }]}>
                {directionArrow(direction)}
              </Text>
            </Animated.View>
          )}
        </View>

        {/* ── Reticle + detection box + proximity ring ───────────────── */}
        <View style={styles.reticleArea}>

          {/* Proximity ring — pulses faster as object gets closer */}
          {isRunning && (
            <Animated.View style={[styles.proxRing, proxRingStyle()]} pointerEvents="none" />
          )}

          <View style={styles.reticle}>
            <View style={[styles.rc, styles.rcTl, detection && styles.rcActive]} />
            <View style={[styles.rc, styles.rcTr, detection && styles.rcActive]} />
            <View style={[styles.rc, styles.rcBl, detection && styles.rcActive]} />
            <View style={[styles.rc, styles.rcBr, detection && styles.rcActive]} />
            <View style={[styles.reticleDot, detection && styles.reticleDotActive]} />
            <Animated.View style={[
              styles.reticleScan,
              {
                opacity: scanPulse.interpolate({ inputRange: [0, 1], outputRange: [0.9, 0.15] }),
                transform: [{ translateY: scanPulse.interpolate({ inputRange: [0, 1], outputRange: [8, 236] }) }],
              },
            ]} />
          </View>

          {/* Direction arrow overlay — shown when guidance is active */}
          {direction && direction !== Direction.LOCKED && (
            <Animated.View style={[styles.dirOverlay, { opacity: dirFade }]}>
              <Text style={[styles.dirOverlayArrow, { color: dirArrowColor() }]}>
                {directionArrow(direction)}
              </Text>
              <Text style={styles.dirOverlayLabel}>{directionLabel(direction)}</Text>
            </Animated.View>
          )}

          {/* Detection bounding box */}
          {detection && (
            <View style={[
              styles.detectionBox,
              {
                left:   `${detection.bbox.x * 100}%`,
                top:    `${detection.bbox.y * 100}%`,
                width:  `${detection.bbox.width * 100}%`,
                height: `${detection.bbox.height * 100}%`,
              },
            ]}>
              <Text style={styles.detectionLabel}>
                {detection.displayLabel || detection.label}  {Math.round((detection.confidence ?? 0) * 100)}%
              </Text>
            </View>
          )}

          {/* Start button (idle) */}
          {!cameraActive && !isRunning && (
            <Pressable style={styles.startButton} onPress={() => setCameraActive(true)}>
              <Ionicons name="camera-outline" size={30} color="#03100c" />
              <Text style={styles.startButtonText}>Start</Text>
            </Pressable>
          )}
        </View>

        {/* ── Bottom target bar ─────────────────────────────────────── */}
        <View style={styles.targetBar}>
          <Pressable
            style={[styles.micButton, micPrimed && styles.micButtonActive]}
            onPress={() => { setMicPrimed(true); inputRef.current?.focus(); }}
            accessibilityLabel="Tap to dictate a target object"
          >
            <Ionicons name="mic-outline" size={22} color="#ffffff" />
          </Pressable>

          <View style={styles.targetDisplay}>
            <TextInput
              ref={inputRef}
              value={target}
              onChangeText={t => { setTarget(t); if (micPrimed && t.trim()) setMicPrimed(false); }}
              placeholder="say or type what to find…"
              placeholderTextColor="rgba(255,255,255,0.62)"
              style={styles.input}
              autoCorrect={false}
              returnKeyType="search"
              onSubmitEditing={canScan ? startScan : undefined}
            />
            <Pressable
              style={[styles.goButton, !canScan && styles.disabled]}
              onPress={canScan ? startScan : null}
              disabled={!canScan}
            >
              <Text style={styles.goButtonText}>Go</Text>
            </Pressable>
          </View>

          <Pressable style={styles.scanButton} onPress={handleScanButton}>
            <Ionicons
              name={isRunning ? 'square' : 'scan-outline'}
              size={isRunning ? 18 : 21}
              color="#03100c"
            />
          </Pressable>
        </View>
      </SafeAreaView>
    </View>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <AppErrorBoundary>
      <AppContent />
    </AppErrorBoundary>
  );
}

// ── Speech helpers ────────────────────────────────────────────────────────────

function stageForGuidance(state, target) {
  const name = state.detection?.displayLabel || state.detection?.label || target?.trim() || 'the target';
  const distance = state.distanceText || (state.distanceMeters != null ? `${state.distanceMeters.toFixed(1)} meters` : null);
  const fallback = state.sentence || `Looking for ${name}.`;
  const stages = {
    idle: { label: 'Ready', instruction: target?.trim() ? `Ready to look for ${name}.` : 'Enter a target to begin.' },
    loading: { label: 'Checking', instruction: 'Checking the configured vision service.' },
    looking: { label: 'Looking', instruction: fallback },
    locked: { label: 'Target centered', instruction: fallback },
    closer: { label: 'Move closer', instruction: fallback },
    lost: { label: 'Target lost', instruction: fallback },
    reacquiring: { label: 'Reacquiring', instruction: fallback },
    reach: { label: 'Target centered', instruction: 'The target appears centered. Reach is disabled until sensing is validated.' },
    error: { label: 'Unavailable', instruction: state.error || 'Vision is unavailable. Check the server and retry.' },
  };
  return { ...(stages[state.status] || stages.looking), distanceText: distance };
}

function _buildSpeech(stageKey, stage, target) {
  const t = target?.trim() ? ` for ${target.trim()}` : '';
  const d = stage?.distanceText ? `. Distance ${stage.distanceText}.` : '';

  if (stageKey === 'loading') return `Checking the vision service${t}.`;
  if (stageKey === 'looking') return `Looking${t}. ${stage?.instruction ?? ''}`;
  if (stageKey === 'locked') return `Target centered${t}. ${stage?.instruction ?? ''}${d}`;
  if (stageKey === 'closer') return `Move closer${t}. ${stage?.instruction ?? ''}${d}`;
  if (stageKey === 'lost' || stageKey === 'reacquiring') return `${stage?.label ?? 'Target lost'}. ${stage?.instruction ?? ''}`;
  if (stageKey === 'error') return `Vision unavailable. ${stage?.instruction ?? ''}`;
  return `${stage?.label ?? ''}. ${stage?.instruction ?? ''}${d}`;
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  app:         { flex: 1, backgroundColor: '#020504' },
  camera:      { ...StyleSheet.absoluteFillObject },
  cameraShade: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(2,5,4,0.16)' },

  scanSweep: {
    position: 'absolute', left: 0, right: 0, top: 0,
    height: 1.5, zIndex: 2,
    backgroundColor: 'rgba(0,255,157,0.55)',
    shadowColor: '#00ff9d', shadowOpacity: 0.85, shadowRadius: 14,
  },

  overlay: { flex: 1, position: 'relative' },

  // Status strip
  signalStrip: {
    position: 'absolute', left: 14, top: 14, zIndex: 8, maxWidth: '70%',
    paddingHorizontal: 12, paddingVertical: 10,
    borderRadius: 14, backgroundColor: 'rgba(2,5,4,0.68)',
  },
  signalHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  signalStatus: { color: '#fff', fontSize: 13, fontWeight: '900', textTransform: 'uppercase' },
  modeBadge:    { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 6 },
  modeBadgeServer: { backgroundColor: 'rgba(255,193,7,0.32)' },
  modeBadgeSimulation: { backgroundColor: 'rgba(255,165,0,0.42)' },
  modeBadgeUnavailable: { backgroundColor: 'rgba(239,71,111,0.42)' },
  modeBadgeText:{ color: '#fff', fontSize: 9, fontWeight: '900', letterSpacing: 1 },
  signalDetail: { marginTop: 2, color: 'rgba(255,255,255,0.85)', fontSize: 12, fontWeight: '700', lineHeight: 16 },
  latencyText:  { marginTop: 2, color: 'rgba(255,255,255,0.45)', fontSize: 10, fontWeight: '600' },
  healthText: { marginTop: 6, color: '#ffd166', fontSize: 11, fontWeight: '700', lineHeight: 15 },
  healthActions: { gap: 7, marginTop: 2 },
  retryButton: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 7, borderRadius: 8, backgroundColor: '#118ab2' },
  demoButton: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 7, borderRadius: 8, backgroundColor: 'rgba(255,165,0,0.55)' },
  retryButtonText: { color: '#fff', fontSize: 10, fontWeight: '900' },

  // Top-right rail
  topRail: {
    position: 'absolute', right: 14, top: 14, zIndex: 8,
    alignItems: 'flex-end', gap: 8,
  },
  sensorPill: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    paddingHorizontal: 12, height: 42, borderRadius: 12,
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.14)',
    backgroundColor: 'rgba(2,5,4,0.62)',
  },
  sensorPillText: { color: '#fff', fontWeight: '800', fontSize: 14 },

  // Direction badge (top-right, below compass)
  dirBadge: {
    alignItems: 'center', justifyContent: 'center',
    width: 52, height: 52, borderRadius: 14,
    backgroundColor: 'rgba(2,5,4,0.72)',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.16)',
  },
  dirArrow: { fontSize: 26, fontWeight: '900' },

  // Reticle area
  reticleArea: { flex: 1, alignItems: 'center', justifyContent: 'center', position: 'relative' },

  // Proximity ring
  proxRing: {
    position: 'absolute',
    width: 310, height: 310,
    borderRadius: 155,
    borderWidth: 2.5,
  },

  // Corner reticle
  reticle: { position: 'relative', width: 260, height: 260 },
  rc: {
    position: 'absolute', width: 28, height: 28,
    borderColor: 'rgba(255,255,255,0.5)',
  },
  rcTl: { top: 0, left:  0, borderTopWidth: 2, borderLeftWidth:  2 },
  rcTr: { top: 0, right: 0, borderTopWidth: 2, borderRightWidth: 2 },
  rcBl: { bottom: 0, left:  0, borderBottomWidth: 2, borderLeftWidth:  2 },
  rcBr: { bottom: 0, right: 0, borderBottomWidth: 2, borderRightWidth: 2 },
  rcActive: { borderColor: '#00ff9d' },
  reticleDot: {
    position: 'absolute', left: '50%', top: '50%',
    width: 7, height: 7, marginLeft: -3.5, marginTop: -3.5,
    borderRadius: 4, backgroundColor: 'rgba(255,255,255,0.45)',
  },
  reticleDotActive: { backgroundColor: '#00ff9d' },
  reticleScan: {
    position: 'absolute', left: 8, right: 8, top: 0,
    height: 1.5, backgroundColor: 'rgba(255,255,255,0.45)',
  },

  // Direction overlay (centre of reticle)
  dirOverlay: {
    position: 'absolute',
    bottom: '12%',
    alignItems: 'center',
    gap: 4,
  },
  dirOverlayArrow: {
    fontSize: 60,
    fontWeight: '900',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 6,
  },
  dirOverlayLabel: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: 13,
    fontWeight: '800',
    textTransform: 'uppercase',
    letterSpacing: 1.5,
  },

  // Detection box
  detectionBox: {
    position: 'absolute',
    borderWidth: 3, borderColor: '#00ff9d',
    borderRadius: 3, backgroundColor: 'rgba(0,255,157,0.06)',
  },
  detectionLabel: {
    position: 'absolute', top: -28, left: 0,
    paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8, overflow: 'hidden',
    color: '#101917', backgroundColor: '#00ff9d',
    fontSize: 11, fontWeight: '900', textTransform: 'capitalize',
  },

  // Start button
  startButton: {
    position: 'absolute', left: '50%', top: '50%',
    zIndex: 7, alignItems: 'center', justifyContent: 'center', gap: 8,
    width: 128, height: 128, marginLeft: -64, marginTop: -64,
    borderRadius: 64, backgroundColor: '#00ff9d',
  },
  startButtonText: { color: '#03100c', fontSize: 16, fontWeight: '900' },

  // Bottom target bar
  targetBar: {
    position: 'absolute', left: 14, right: 14, bottom: 16, zIndex: 9,
    flexDirection: 'row', alignItems: 'center', gap: 10,
    padding: 10, borderWidth: 1, borderColor: 'rgba(255,255,255,0.16)',
    borderRadius: 18, backgroundColor: 'rgba(2,5,4,0.80)',
  },
  micButton: {
    alignItems: 'center', justifyContent: 'center',
    width: 52, height: 52, borderRadius: 12,
    backgroundColor: 'rgba(255,255,255,0.12)',
  },
  micButtonActive: {
    backgroundColor: 'rgba(0,255,157,0.22)',
    borderWidth: 1, borderColor: 'rgba(0,255,157,0.55)',
  },
  targetDisplay: {
    flex: 1, flexDirection: 'row', alignItems: 'center', gap: 8,
    minWidth: 0, height: 52, paddingLeft: 14, paddingRight: 6,
    borderRadius: 12, backgroundColor: 'rgba(255,255,255,0.09)',
  },
  input: { flex: 1, minWidth: 0, color: '#fff', fontSize: 16, fontWeight: '800' },
  goButton: {
    alignItems: 'center', justifyContent: 'center',
    minWidth: 46, height: 46, paddingHorizontal: 12,
    borderRadius: 10, backgroundColor: '#00ff9d',
  },
  goButtonText: { color: '#03100c', fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
  scanButton: {
    alignItems: 'center', justifyContent: 'center',
    width: 52, height: 52, borderRadius: 12, backgroundColor: '#00ff9d',
  },
  disabled: { opacity: 0.35 },

  // Permission screens
  permissionScreen: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    gap: 16, padding: 26, backgroundColor: '#101917',
  },
  permissionTitle: { color: '#fff', fontSize: 27, fontWeight: '900', textAlign: 'center' },
  permissionText:  { color: '#aab7b4', fontSize: 16, lineHeight: 23, textAlign: 'center' },
  primaryButton:   {
    alignItems: 'center', justifyContent: 'center',
    minHeight: 54, minWidth: 190, paddingHorizontal: 18,
    borderRadius: 16, backgroundColor: '#118ab2',
  },
  primaryButtonText: { color: '#fff', fontSize: 16, fontWeight: '900' },
});
