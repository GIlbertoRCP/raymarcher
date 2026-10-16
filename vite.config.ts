import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const page = (file: string) => fileURLToPath(new URL(`./web/${file}`, import.meta.url));

export default defineConfig({
  root: 'web',
  // Relative asset paths so the site works under a sub-path (GitHub Pages).
  base: './',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        plates: page('index.html'),
        verification: page('validation.html'),
      },
    },
  },
  test: {
    root: '.',
    include: ['web/src/**/*.test.ts', 'compiler/test/**/*.test.ts', 'validation/**/*.test.ts'],
  },
});
