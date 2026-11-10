import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const base = process.argv[2] || 'http://127.0.0.1:4321/Rabbit';
const outDir = resolve(process.cwd(), '.preview');
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({ channel: 'chrome' });

async function capture(name, path) {
  for (const scheme of ['light', 'dark']) {
    const context = await browser.newContext({
      viewport: { width: 1280, height: 900 },
      deviceScaleFactor: 2,
      colorScheme: scheme,
      reducedMotion: 'reduce',
    });
    const page = await context.newPage();
    const url = path === '/' ? `${base}/` : `${base}${path}`;
    await page.goto(url, { waitUntil: 'networkidle', timeout: 45000 });
    await page.waitForTimeout(500);
    const file = resolve(outDir, `${name}-1280-${scheme}.png`);
    await page.screenshot({ path: file, fullPage: true });
    console.log(file);
    await context.close();
  }
}

await capture('benchmarks', '/benchmarks');
await capture('roadmap', '/roadmap');
await capture('download', '/download');
await capture('faq', '/faq');
await capture('changelog', '/changelog');
await capture('docs', '/docs');

await browser.close();
