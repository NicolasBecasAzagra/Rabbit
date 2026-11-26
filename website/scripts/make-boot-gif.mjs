// Builds the animated boot log shown in the README.
//
//   npm run gif -- [--out ../docs/boot.gif] [--frames /some/empty/dir]
//
// The lines are read from the hero of the site (Hero.astro) and checked against
// docs/BOOT.md: each one must be a whole line of the recorded log, in the same
// order, or the script stops. No dev server is needed. Every frame is rendered
// from an explicit state with an explicit duration, so the GIF is
// deterministic. The typing speed and the pauses are presentation only; they
// are not the timing of the real boot.
// Needs ffmpeg on the PATH and a local Google Chrome.
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};

const out = resolve(process.cwd(), arg('out', join(root, '../docs/boot.gif')));
const framesDir = arg('frames')
  ? resolve(process.cwd(), arg('frames'))
  : mkdtempSync(join(tmpdir(), 'rabbit-boot-gif-'));
mkdirSync(framesDir, { recursive: true });

// Logical size, close to the width of a README column, and the pixel ratio.
const width = 880;
const height = 516;
const scale = 1.5;

// --- The lines, from the source of the site, verified against the record. ---
const hero = readFileSync(join(root, 'src/components/Hero.astro'), 'utf8');
const block = hero.match(/const bootLog = \[([\s\S]*?)\];/);
if (!block) throw new Error('bootLog array not found in Hero.astro');
const lines = [...block[1].matchAll(/'([^'\\]*)'/g)].map((m) => m[1]);
if (lines.length === 0) throw new Error('bootLog array is empty');

const bootDoc = readFileSync(join(root, '../docs/BOOT.md'), 'utf8');
const record = bootDoc.match(/## Runtime sequence[\s\S]*?```text\n([\s\S]*?)```/);
if (!record) throw new Error('runtime sequence not found in docs/BOOT.md');
const recorded = record[1].split('\n');
let cursor = 0;
for (const line of lines) {
  const at = recorded.indexOf(line, cursor);
  if (at < 0) throw new Error(`not in docs/BOOT.md, or out of order: ${line}`);
  cursor = at + 1;
}

// Lines that get a marker and a longer pause (milliseconds).
const holds = new Map([
  ['Rabbit: x86_64 kernel, M4b (SMP)', 700],
  ['Hello from Rabbit userland', 900],
  ['USER OK exit=0', 800],
  ['Rabbit: exception survived', 1200],
  ['Rabbit: boot OK', 2000],
]);
const finalLine = lines.at(-1);

// --- The page. ---
const font = (file) =>
  `url("data:font/woff2;base64,${readFileSync(join(root, 'public/fonts', file)).toString('base64')}")`;
const logo = readFileSync(join(root, 'public/logo.svg'), 'utf8');
const ears = logo.match(/<path d="([^"]+)"/)?.[1];
if (!ears) throw new Error('mark not found in public/logo.svg');
const mark = (cls) =>
  `<svg class="${cls}" viewBox="9 2 14 28" aria-hidden="true"><path d="${ears}"/></svg>`;

const escape = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
@font-face { font-family: Inter; src: ${font('inter-latin.woff2')}; font-weight: 100 900; }
@font-face { font-family: Mono; src: ${font('jetbrains-mono-latin.woff2')}; font-weight: 100 800; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { width: ${width}px; height: ${height}px; background: #000; color: #fff; font-family: Mono; font-variant-ligatures: none; }
.window { position: relative; width: 100%; height: 100%; display: flex; flex-direction: column; border: 2px solid #6f4528; }
header { height: 56px; flex: none; display: flex; align-items: center; gap: 12px; padding: 0 28px; background: #6f4528; }
header .small { width: 12px; height: 24px; fill: #fff; }
header .name { font-family: Inter; font-weight: 800; font-size: 22px; letter-spacing: -0.03em; }
header .port { margin-left: auto; font-size: 16px; letter-spacing: 0.08em; text-transform: uppercase; }
ol { list-style: none; flex: 1; padding: 18px 14px 0; }
li { height: 38px; display: flex; align-items: center; padding: 0 14px; font-size: 22px; white-space: pre; color: #ede4dc; }
li .square { width: 11px; height: 11px; flex: none; margin-right: 16px; }
li.key { color: #fff; font-weight: 700; }
li.key.done .square { background: #c49a7c; }
li.final.done { background: #6f4528; }
li.final.done .square { background: #fff; }
.cursor { display: inline-block; width: 12px; height: 24px; margin-left: 3px; background: #fff; }
footer { height: 48px; flex: none; display: flex; align-items: center; justify-content: space-between; margin: 0 28px; border-top: 1px solid #4a3b32; font-size: 16px; color: #c9b9ad; }
.end:not(.logo) > * { visibility: hidden; }
.end { position: absolute; inset: 56px 0 48px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #fff; color: #000; }
.lockup { display: flex; align-items: flex-end; gap: 34px; }
.lockup .big { width: 100px; height: 200px; fill: #000; }
.word { font-family: Inter; font-weight: 800; font-size: 150px; line-height: 0.76; letter-spacing: -0.06em; }
.verdict { margin-top: 44px; display: flex; align-items: center; gap: 16px; padding: 12px 24px; background: #6f4528; color: #fff; font-size: 24px; font-weight: 700; }
.verdict i { width: 12px; height: 12px; background: #fff; }
</style></head><body><div class="window">
<header>${mark('small')}<span class="name">Rabbit</span><span class="port">serial0</span></header>
<ol></ol>
<footer><span>serial output under QEMU</span><span>excerpt of docs/BOOT.md</span></footer>
<div class="end" hidden><div class="lockup">${mark('big')}<span class="word">Rabbit</span></div>
<p class="verdict"><i></i>${escape(finalLine)}</p></div>
</div></body></html>`;

// --- The storyboard: a list of states, each with its duration in ms. ---
// GIF delays are counted in centiseconds, so durations are multiples of 10 ms.
const states = [];
const add = (state, ms) => states.push({ state, ms });

add({ shown: 0, chars: 0, cursor: true }, 400);
add({ shown: 0, chars: 0, cursor: false }, 280);
lines.forEach((line, index) => {
  const step = 3;
  for (let chars = step; chars < line.length; chars += step) {
    add({ shown: index, chars, cursor: true }, 40);
  }
  const hold = holds.get(line);
  add({ shown: index + 1, chars: 0, cursor: line !== finalLine }, hold ?? 240);
});
// Closing frame: a short fade, then a long rest before the loop restarts.
const full = { shown: lines.length, chars: 0, cursor: false };
// The white panel arrives empty, so the logo never sits on top of the log.
for (const fade of [0.35, 0.7, 1]) add({ ...full, fade }, 80);
add({ ...full, fade: 1, logo: true }, 3600);

const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: scale });
await page.setContent(html, { waitUntil: 'load' });
await page.evaluate(() =>
  Promise.all([document.fonts.load('22px Mono'), document.fonts.load('800 150px Inter')])
);

const frameName = (i) => `frame-${String(i).padStart(4, '0')}.png`;
for (const [i, { state }] of states.entries()) {
  await page.evaluate(
    ({ state, lines, keys, finalLine }) => {
      const list = document.querySelector('ol');
      list.replaceChildren();
      const typing = state.chars > 0;
      const count = state.shown + (typing || state.shown === 0 ? 1 : 0);
      for (let n = 0; n < count; n += 1) {
        const done = n < state.shown;
        const li = document.createElement('li');
        if (keys.includes(lines[n])) li.classList.add('key');
        if (lines[n] === finalLine) li.classList.add('final');
        if (done) li.classList.add('done');
        const square = document.createElement('span');
        square.className = 'square';
        li.append(square, done ? lines[n] : lines[n].slice(0, state.chars));
        if (state.cursor && n === count - 1) {
          const caret = document.createElement('span');
          caret.className = 'cursor';
          li.append(caret);
        }
        list.append(li);
      }
      const end = document.querySelector('.end');
      end.hidden = !state.fade;
      end.classList.toggle('logo', Boolean(state.logo));
      end.style.opacity = String(state.fade ?? 0);
    },
    { state, lines, keys: [...holds.keys()], finalLine }
  );
  // A line that does not fit would be cut off in the GIF without any error.
  const overflow = await page.evaluate(() =>
    [...document.querySelectorAll('li')].some((li) => li.scrollWidth > li.clientWidth)
  );
  if (overflow) throw new Error(`a line overflows the frame in state ${i}`);
  await page.screenshot({ path: join(framesDir, frameName(i)) });
}
await browser.close();

// The concat demuxer carries the duration of each frame. It ignores the one of
// the last entry, so the muxer receives it as the final delay.
const concat = join(framesDir, 'frames.txt');
writeFileSync(
  concat,
  states.map(({ ms }, i) => `file '${frameName(i)}'\nduration ${ms / 1000}\n`).join('')
);

mkdirSync(dirname(out), { recursive: true });
execFileSync(
  'ffmpeg',
  [
    '-y', '-loglevel', 'error',
    '-f', 'concat', '-safe', '0', '-i', concat,
    '-vf', 'split[a][b];[a]palettegen=max_colors=64:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle',
    '-fps_mode', 'passthrough',
    '-final_delay', String(states.at(-1).ms / 10),
    '-loop', '0',
    out,
  ],
  { stdio: 'inherit' }
);

console.log(
  JSON.stringify(
    {
      out,
      framesDir,
      lines: lines.length,
      frames: states.length,
      seconds: states.reduce((sum, s) => sum + s.ms, 0) / 1000,
      size: `${Math.round(width * scale)}x${Math.round(height * scale)}`,
      bytes: statSync(out).size,
    },
    null,
    2
  )
);
