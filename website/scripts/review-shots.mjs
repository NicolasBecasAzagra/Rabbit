import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const base = process.argv[2] || 'http://127.0.0.1:4321/Rabbit';
const outDir = resolve(process.cwd(), '.preview');
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({ channel: 'chrome' });

async function capture(name, path, options = {}) {
  for (const scheme of ['light', 'dark']) {
    for (const width of options.widths || [1280]) {
      const context = await browser.newContext({
        viewport: { width, height: width < 500 ? 812 : 900 },
        deviceScaleFactor: 2,
        colorScheme: scheme,
        reducedMotion: 'reduce',
      });
      const page = await context.newPage();
      const url = path === '/' ? `${base}/` : `${base}${path}`;
      await page.goto(url, { waitUntil: 'networkidle', timeout: 45000 });
      await page.waitForTimeout(500);
      if (options.selector) {
        await page.locator(options.selector).scrollIntoViewIfNeeded();
        await page.waitForTimeout(300);
      }
      const file = resolve(outDir, `${name}-${width}-${scheme}.png`);
      await page.screenshot({ path: file, fullPage: !!options.fullPage });
      console.log(file);
      await context.close();
    }
  }
}

await capture('home', '/', { widths: [1280], fullPage: true });
await capture('home-mobile', '/', { widths: [375], fullPage: false });
await capture('layers', '/', { widths: [1280], selector: '.layers-section', fullPage: false });

await browser.close();
