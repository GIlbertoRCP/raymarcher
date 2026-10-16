// Opens the WebGPU app in the installed Chrome (playwright-core) on a private
// Vite dev server, for the validation and benchmark scripts.
import { join } from 'node:path';
import { chromium, type Page } from 'playwright-core';
import { createServer } from 'vite';

export interface AppOptions {
  root: string;
  headed: boolean;
  channel: string;
}

export function appOptions(root: string, args: string[]): AppOptions {
  const i = args.indexOf('--channel');
  return { root, headed: args.includes('--headed'), channel: i >= 0 ? args[i + 1] : 'chrome' };
}

/** Runs fn with a page that has the dev-only hooks (window.raymarcherCapture, window.raymarcherTime). */
export async function withApp<T>(opts: AppOptions, fn: (page: Page) => Promise<T>): Promise<T> {
  const server = await createServer({
    configFile: join(opts.root, 'vite.config.ts'),
    server: { port: 5174, strictPort: true },
    logLevel: 'error',
  });
  await server.listen();
  const browser = await chromium.launch({
    channel: opts.channel,
    headless: !opts.headed,
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.on('pageerror', e => console.error('page error:', e.message));
    await page.goto('http://localhost:5174/');
    await page.waitForFunction(() => 'raymarcherCapture' in window, null, { timeout: 30_000 });
    const hasGpu = await page.evaluate(async () => Boolean(navigator.gpu && (await navigator.gpu.requestAdapter())));
    if (!hasGpu) throw new Error(`No WebGPU adapter in ${opts.headed ? 'headed' : 'headless'} ${opts.channel}; try --headed`);
    return await fn(page);
  } finally {
    await browser.close();
    await server.close();
  }
}
