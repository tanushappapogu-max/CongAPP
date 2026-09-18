import { describe, expect, test } from 'vitest';
import {
  GUIDANCE_REPLAY_FIXTURES,
  createInitialGuidanceState,
  guidanceReducer,
} from '../../../packages/pulsepoint-core/src/index.js';

function replay(fixture) {
  let state = createInitialGuidanceState(fixture.target);
  const states = [];
  for (const event of fixture.events) {
    state = guidanceReducer(state, event, fixture.options);
    states.push(state);
  }
  return { final: state, states };
}

describe('shared guidance replay contract', () => {
  test('includes every required deterministic scenario', () => {
    expect(GUIDANCE_REPLAY_FIXTURES.map(({ name }) => name)).toEqual([
      'found-locked',
      'lost',
      'reacquired',
      'moving-target',
      'multiple-candidates',
      'low-light-safe',
      'low-confidence-safe',
      'reach-policy',
    ]);
  });

  test.each(GUIDANCE_REPLAY_FIXTURES)('$name replays to its contract state', (fixture) => {
    const first = replay(fixture);
    expect(first.final).toMatchObject(fixture.expected.final);
    for (const checkpoint of fixture.expected.checkpoints || []) {
      expect(first.states[checkpoint.after - 1]).toMatchObject(checkpoint.state);
    }
    expect(replay(fixture)).toEqual(first);
  });
});
