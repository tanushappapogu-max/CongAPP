// Prompt pack: precomputed YOLOE text embeddings (built by scripts/yoloe/build_prompt_pack.py).
// A search feeds YOLOE the target's vector plus its look-alike negatives, so a box only counts
// as the target when it scores higher for the target than for anything it could be confused with.

import { normalizeTargetText, stringSimilarity, resolveCocoTarget } from './coco.js';
import { REFERENCE_WIDTHS_CM } from './distance.js';

const PACK_JSON_URL = '/prompts/pack.json';
const PACK_BIN_URL = '/prompts/pack.bin';
const FUZZY_THRESHOLD = 0.8;

let _pack = null;
let _loadPromise = null;

export function parsePromptPack(meta, buffer) {
  const dim = meta.dim;
  const vectors = new Float32Array(buffer);
  if (vectors.length !== meta.items.length * dim) {
    throw new Error(`Prompt pack size mismatch: ${vectors.length} floats for ${meta.items.length} items`);
  }
  const byName = new Map();
  const byPhrase = new Map();
  meta.items.forEach((raw, index) => {
    const item = {
      name: raw.name,
      aliases: raw.aliases || [],
      negatives: raw.negatives || [],
      widthCm: raw.widthCm ?? REFERENCE_WIDTHS_CM[raw.name] ?? null,
      index,
    };
    byName.set(item.name, item);
    for (const phrase of [item.name, ...item.aliases]) {
      const key = normalizeTargetText(phrase);
      if (key && !byPhrase.has(key)) byPhrase.set(key, item);
    }
  });
  return { dim, vectors, byName, byPhrase, model: meta.model };
}

export function setPromptPack(pack) {
  _pack = pack;
}

export function isPromptPackReady() {
  return _pack !== null;
}

export function loadPromptPack({ signal } = {}) {
  if (_pack) return Promise.resolve(_pack);
  if (!_loadPromise) {
    _loadPromise = Promise.all([
      fetch(PACK_JSON_URL, { signal }).then(r => {
        if (!r.ok) throw new Error(`Failed to fetch prompt pack: ${r.status}`);
        return r.json();
      }),
      fetch(PACK_BIN_URL, { signal }).then(r => {
        if (!r.ok) throw new Error(`Failed to fetch prompt vectors: ${r.status}`);
        return r.arrayBuffer();
      }),
    ]).then(([meta, buffer]) => {
      _pack = parsePromptPack(meta, buffer);
      return _pack;
    }).finally(() => {
      _loadPromise = null;
    });
  }
  return _loadPromise;
}

/**
 * Map what the user said or typed to a pack item.
 * @returns {{ item: object, score: number, source: 'exact'|'coco'|'fuzzy' } | null}
 */
export function resolvePromptTarget(text, pack = _pack) {
  if (!pack) return null;
  const norm = normalizeTargetText(text);
  if (!norm) return null;

  const exact = pack.byPhrase.get(norm) || pack.byPhrase.get(norm.replace(/^(my|the|a|an) /, ''));
  if (exact) return { item: exact, score: 1, source: 'exact' };

  const coco = resolveCocoTarget(norm);
  if (coco && pack.byName.has(coco)) return { item: pack.byName.get(coco), score: 1, source: 'coco' };

  let best = null;
  for (const [phrase, item] of pack.byPhrase) {
    const score = Math.max(stringSimilarity(norm, phrase), editSimilarity(norm, phrase));
    if (!best || score > best.score) best = { item, score };
  }
  return best && best.score >= FUZZY_THRESHOLD ? { ...best, source: 'fuzzy' } : null;
}

function editSimilarity(a, b) {
  const longest = Math.max(a.length, b.length);
  if (!longest) return 0;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return 1 - prev[b.length] / longest;
}

/** Target first, then its negatives. Returns names plus a flat [K * dim] Float32Array for the model. */
export function buildPromptSet(item, pack = _pack) {
  const members = [item, ...item.negatives.map(n => pack.byName.get(n)).filter(Boolean)];
  const data = new Float32Array(members.length * pack.dim);
  members.forEach((m, k) => {
    data.set(pack.vectors.subarray(m.index * pack.dim, (m.index + 1) * pack.dim), k * pack.dim);
  });
  return { key: item.name, names: members.map(m => m.name), dim: pack.dim, data };
}
