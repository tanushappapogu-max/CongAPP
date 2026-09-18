import Constants from 'expo-constants';
import { normalizeDetection } from '../../../packages/pulsepoint-core/src/index.js';

const DEFAULT_TIMEOUT_MS = 5500;
const MAX_UPLOAD_QUALITY = 0.45;

function readExtraUrl() {
  const extra = Constants?.expoConfig?.extra || {};
  return extra.pulsepointVisionApiUrl || extra.PULSEPOINT_VISION_API_URL || extra.visionApiUrl || null;
}

function validateUrl(value) {
  if (typeof value !== 'string' || !value.trim() || /\s/.test(value)) return null;
  try {
    const url = new URL(value.trim());
    if (!['http:', 'https:'].includes(url.protocol) || !url.hostname) return null;
    return url.toString().replace(/\/+$/, '');
  } catch {
    return null;
  }
}

const initialUrl = validateUrl(process.env.EXPO_PUBLIC_PULSEPOINT_VISION_API_URL) || validateUrl(readExtraUrl());
let _serverUrl = initialUrl;

export const configuredVisionApiUrl = initialUrl;

export function setServerUrl(url) {
  const validated = validateUrl(url);
  if (!validated) throw new Error('Invalid Pulse Point Vision API URL. Use an http(s) URL.');
  _serverUrl = validated;
  return _serverUrl;
}

export function getServerUrl() { return _serverUrl; }
export function isVisionApiConfigured() { return Boolean(_serverUrl); }

// Simulation is never selected by a health failure. This flag only makes a
// future explicit demo mode possible and is intentionally opt-in.
export function isSimulationOptedIn() {
  const extra = Constants?.expoConfig?.extra || {};
  return process.env.EXPO_PUBLIC_PULSEPOINT_ALLOW_SIMULATION === 'true'
    || extra.pulsepointAllowSimulation === true;
}

function unavailableError() {
  const error = new Error('Vision API URL is not configured. Set EXPO_PUBLIC_PULSEPOINT_VISION_API_URL or Expo extra.pulsepointVisionApiUrl.');
  error.code = 'VISION_API_NOT_CONFIGURED';
  return error;
}

async function fetchWithTimeout(path, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  if (!_serverUrl) throw unavailableError();
  const controller = new AbortController();
  const callerSignal = options.signal;
  let timedOut = false;
  const onCallerAbort = () => controller.abort();
  if (callerSignal) {
    if (callerSignal.aborted) controller.abort();
    else callerSignal.addEventListener('abort', onCallerAbort, { once: true });
  }
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    return await fetch(`${_serverUrl}${path}`, { ...options, signal: controller.signal });
  } catch (error) {
    if (timedOut) {
      const timeout = new Error(`Vision API request timed out after ${timeoutMs} ms.`);
      timeout.name = 'TimeoutError';
      timeout.code = 'VISION_API_TIMEOUT';
      throw timeout;
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
    callerSignal?.removeEventListener('abort', onCallerAbort);
  }
}

function metadata(path, response, startedAt, extra = {}) {
  return {
    url: `${_serverUrl || ''}${path}`,
    endpoint: path,
    status: response?.status ?? null,
    latencyMs: Date.now() - startedAt,
    source: 'server',
    ...extra,
  };
}

function readJson(response) { return response.json().catch(() => ({})); }

export async function detectObject(imageUri, targetName, { signal, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const formData = new FormData();
  formData.append('image', { uri: imageUri, type: 'image/jpeg', name: 'frame.jpg' });
  if (targetName) formData.append('target', targetName.trim().toLowerCase());
  const startedAt = Date.now();
  const response = await fetchWithTimeout('/detect', {
    method: 'POST',
    body: formData,
    headers: { Accept: 'application/json' },
    signal,
  }, timeoutMs);
  const data = await readJson(response);
  const meta = metadata('/detect', response, startedAt, {
    upload: { quality: MAX_UPLOAD_QUALITY, format: 'image/jpeg', maxBytes: 10 * 1024 * 1024 },
  });
  if (!response.ok) {
    const error = new Error(`Vision API error: ${response.status}`);
    error.code = 'VISION_API_HTTP_ERROR';
    error.status = response.status;
    error.meta = meta;
    throw error;
  }
  return { data, meta };
}

export async function checkHealth({ signal, timeoutMs = 3000 } = {}) {
  const startedAt = Date.now();
  if (!_serverUrl) return { ok: false, data: null, meta: metadata('/health', null, startedAt), error: unavailableError() };
  try {
    const response = await fetchWithTimeout('/health', { signal }, timeoutMs);
    const data = await readJson(response);
    const ok = response.ok && data.status === 'ok';
    return { ok, data, meta: metadata('/health', response, startedAt), error: ok ? null : new Error('Vision API health check failed.') };
  } catch (error) {
    return { ok: false, data: null, meta: metadata('/health', null, startedAt), error };
  }
}

export async function listKnownObjects({ signal, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const startedAt = Date.now();
  const response = await fetchWithTimeout('/objects', { signal, headers: { Accept: 'application/json' } }, timeoutMs);
  const data = await readJson(response);
  const meta = metadata('/objects', response, startedAt);
  if (!response.ok) {
    const error = new Error(`Failed to fetch objects: ${response.status}`);
    error.code = 'VISION_API_HTTP_ERROR';
    error.meta = meta;
    throw error;
  }
  return { objects: Array.isArray(data.objects) ? data.objects : [], data, meta };
}

export async function classifyText(text, topK = 3, { signal, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const response = await fetchWithTimeout('/classify-text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ text: text.trim(), top_k: topK }),
    signal,
  }, timeoutMs);
  if (!response.ok) {
    const err = await readJson(response);
    throw new Error(`TextCNN API error ${response.status}: ${err.error || response.statusText}`);
  }
  return response.json();
}

export async function fetchTeaSchema({ signal, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const response = await fetchWithTimeout('/tea-schema', { headers: { Accept: 'application/json' }, signal }, timeoutMs);
  if (!response.ok) throw new Error('Failed to fetch tea schema');
  return response.json();
}

export function apiResultToDetection(apiResponse, targetName, nowMs = Date.now()) {
  const apiResult = apiResponse?.data ?? apiResponse;
  if (!apiResult?.detected || !apiResult.boundingBox) return null;
  const areaEstimate = estimateDistance(apiResult.boundingBox);
  const explicitProof = apiResult.assistiveReady === true
    && (apiResult.assistiveReadyProof === true || apiResult.assistive_ready_proof === true);
  const detection = normalizeDetection({
    label: apiResult.name || targetName || 'target',
    displayLabel: apiResult.name || targetName || 'target',
    bbox: apiResult.boundingBox,
    confidence: apiResult.confidence,
    source: 'server',
    timestampMs: nowMs,
    distance: apiResult.distance || { meters: areaEstimate.meters, method: 'area-estimate', uncertaintyMeters: areaEstimate.uncertaintyMeters },
    assistiveReady: explicitProof,
    model: apiResult.model || apiResult.model_name || null,
  }, { source: 'server', assistiveReady: explicitProof, nowMs });
  return detection ? {
    ...detection,
    apiMeta: { ...(apiResponse?.meta || {}), latencyMs: apiResponse?.meta?.latencyMs ?? apiResult.latency_ms ?? null, source: 'server' },
    alternatives: Array.isArray(apiResult.alternatives) ? apiResult.alternatives : [],
  } : null;
}

function estimateDistance(bbox) {
  const area = Number(bbox?.width) * Number(bbox?.height);
  if (!Number.isFinite(area) || area <= 0) return { meters: null, uncertaintyMeters: null };
  if (area > 0.4) return { meters: 0.3, uncertaintyMeters: 0.5 };
  if (area > 0.2) return { meters: 0.8, uncertaintyMeters: 0.8 };
  if (area > 0.1) return { meters: 1.5, uncertaintyMeters: 1.2 };
  if (area > 0.05) return { meters: 2.5, uncertaintyMeters: 1.8 };
  return { meters: 3.5, uncertaintyMeters: 2.5 };
}
