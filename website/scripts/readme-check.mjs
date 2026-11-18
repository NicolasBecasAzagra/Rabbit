// Checks of src/lib/readme.ts. Run: node --experimental-strip-types scripts/readme-check.mjs
import assert from 'node:assert/strict';
import { loadReadme, parseReadme, plain } from '../src/lib/readme.ts';

assert.equal(plain('**v0.4** [a `b`](http://x)'), 'v0.4 a b');
const r = loadReadme();
console.log('done', r.status.length, '| next', r.next.map((n) => `${n.title} (${n.issues.length})`).join('; '));
assert.ok(r.status.some((s) => s.id === 'M14'));
assert.equal(r.next[0].title, 'v0.4 Credible numbers');
assert.throws(() => parseReadme('# x\n\n## Status\n\n| A | B |\n|---|---|\n| 1 | 2 |\n'), /header/);
assert.throws(() => parseReadme('# x\n'), /no "## Status"/);
console.log('readme-check: all assertions passed');
