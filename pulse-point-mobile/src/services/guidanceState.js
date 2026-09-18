import {
  createInitialGuidanceState,
  EVENTS,
  guidanceReducer,
} from '../../../packages/pulsepoint-core/src/index.js';

export { createInitialGuidanceState, EVENTS };

const EXPERIMENTAL_SOURCES = new Set(['server', 'simulated', 'offline']);

/**
 * Mobile policy wrapper around the shared guidance reducer.
 *
 * The shared reducer owns the state-machine transitions. Mobile adds one
 * safety invariant: an unvalidated detector may show directional guidance,
 * but it can never produce a confident reach signal.
 */
export function mobileGuidanceReducer(state, event, options = {}) {
  const next = guidanceReducer(state, event, options);
  const detection = next?.detection;
  const experimental = EXPERIMENTAL_SOURCES.has(next?.source) || !next?.assistiveReady;

  if (event?.type === EVENTS.DETECTION_RECEIVED && experimental && next?.signal === 'reach') {
    return {
      ...next,
      status: 'closer',
      signal: 'closer',
      proximity: next.proximity === 'reach' ? 'near' : next.proximity,
      sentence: `${detection?.displayLabel || detection?.label || 'Target'} is centered. Experimental detector guidance; do not reach yet.`,
      speechPhrase: 'Closer. Experimental mode.',
      assistiveReady: false,
    };
  }

  return next;
}

