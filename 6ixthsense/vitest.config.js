import { configDefaults, defineConfig } from 'vitest/config';

// coco.test.js (label aliases) is kept but skipped by default; `npm run test:all` includes it.
const skipped = process.env.ALL_TESTS ? [] : ['src/detection/coco.test.js'];

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    exclude: [...configDefaults.exclude, ...skipped],
  },
});
