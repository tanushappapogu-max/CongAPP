import { describe, test, expect } from 'vitest';
import { parsePromptPack, resolvePromptTarget, buildPromptSet } from './prompts.js';

const DIM = 4;
const META = {
  model: 'test',
  dim: DIM,
  items: [
    { name: 'eyeglasses', aliases: ['glasses', 'my glasses', 'reading glasses'], widthCm: 14, negatives: ['sunglasses'] },
    { name: 'sunglasses', aliases: ['shades'], widthCm: 14, negatives: ['eyeglasses'] },
    { name: 'pen', aliases: ['pencil'], widthCm: 1.5, negatives: [] },
    { name: 'wristwatch', aliases: ['watch'], widthCm: 4, negatives: [], coco: 'clock' },
    { name: 'cup', aliases: [], negatives: [] },
    { name: 'couch', aliases: [], negatives: [] },
    { name: 'cell phone', aliases: [], negatives: [] },
  ],
};

function makePack() {
  const vectors = new Float32Array(META.items.length * DIM);
  vectors.forEach((_, i) => { vectors[i] = i; });
  return parsePromptPack(META, vectors.buffer);
}

describe('parsePromptPack', () => {
  test('rejects a vector file that does not match the item count', () => {
    expect(() => parsePromptPack(META, new Float32Array(3).buffer)).toThrow(/mismatch/);
  });

  test('COCO items inherit reference widths from distance.js', () => {
    const pack = makePack();
    expect(pack.byName.get('couch').widthCm).toBe(200);
    expect(pack.byName.get('pen').widthCm).toBe(1.5);
  });
});

describe('resolvePromptTarget', () => {
  const pack = makePack();

  test('returns null without a pack or for empty input', () => {
    expect(resolvePromptTarget('glasses', null)).toBeNull();
    expect(resolvePromptTarget('', pack)).toBeNull();
  });

  test('aliases resolve to the pack item', () => {
    expect(resolvePromptTarget('my glasses', pack)).toMatchObject({ item: { name: 'eyeglasses' }, source: 'exact' });
    expect(resolvePromptTarget('Shades!', pack).item.name).toBe('sunglasses');
  });

  test('leading articles are ignored', () => {
    expect(resolvePromptTarget('the pencil', pack).item.name).toBe('pen');
  });

  test('pen no longer maps to cell phone', () => {
    expect(resolvePromptTarget('pen', pack).item.name).toBe('pen');
  });

  test('falls back to existing COCO aliases', () => {
    expect(resolvePromptTarget('sofa', pack)).toMatchObject({ item: { name: 'couch' }, source: 'coco' });
    expect(resolvePromptTarget('iphone', pack).item.name).toBe('cell phone');
  });

  test('fuzzy matches close spellings but not unrelated words', () => {
    expect(resolvePromptTarget('eyeglases', pack)).toMatchObject({ item: { name: 'eyeglasses' }, source: 'fuzzy' });
    expect(resolvePromptTarget('hovercraft', pack)).toBeNull();
  });
});

describe('buildPromptSet', () => {
  const pack = makePack();

  test('puts the target first followed by its negatives', () => {
    const set = buildPromptSet(pack.byName.get('eyeglasses'), pack);
    expect(set.names).toEqual(['eyeglasses', 'sunglasses']);
    expect(set.dim).toBe(DIM);
    expect(Array.from(set.data)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  test('marks COCO targets so the YOLO11n backup can find them', () => {
    expect(buildPromptSet(pack.byName.get('cup'), pack).cocoLabel).toBe('cup');
    expect(buildPromptSet(pack.byName.get('eyeglasses'), pack).cocoLabel).toBeNull();
    expect(buildPromptSet(pack.byName.get('wristwatch'), pack).cocoLabel).toBe('clock');
  });

  test('copies the right rows for a later item', () => {
    const set = buildPromptSet(pack.byName.get('pen'), pack);
    expect(set.names).toEqual(['pen']);
    expect(Array.from(set.data)).toEqual([8, 9, 10, 11]);
  });
});
