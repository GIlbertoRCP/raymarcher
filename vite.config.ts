import { defineConfig } from 'vitest/config';

export default defineConfig({
  root: 'web',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
  test: {
    root: '.',
    include: ['web/src/**/*.test.ts'],
  },
});
