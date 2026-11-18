// Checks of src/lib/policy.ts. Run: node --experimental-strip-types scripts/policy-check.mjs
import assert from 'node:assert/strict';
import { loadPolicy, parseCsv, parseSummary, parseTotals, verifyTotals } from '../src/lib/policy.ts';

// quoted fields: commas, doubled quotes and line breaks stay inside the field
assert.deepEqual(parseCsv('a,"b,c","d ""e"""\nx,,y\n'), [['a', 'b,c', 'd "e"'], ['x', '', 'y']]);

const p = loadPolicy();
const c = p.counts;
console.log('refused by policy', c.refusedByPolicy, '| linux let through', c.linuxLetThrough, '| unavailable', c.rabbitUnavailable, 'of', c.compromisedCalls);
assert.equal(c.sameOnBothSides + c.differByPolicy + c.differForOtherReasons, c.totalCases);
console.log('cases', c.totalCases, 'same', c.sameOnBothSides, 'policy', c.differByPolicy, 'other', c.differForOtherReasons);
assert.equal(c.linuxLetThrough + c.linuxFailedOnItsOwn, p.headline.length);
assert.equal(p.headline.length, c.refusedByPolicy);
assert.ok(p.headline.every((r) => r.cls === 'policy' && r.rabbit === 'denied'));
// the long quoted note of the family row survives parsing
assert.ok(p.rows.some((r) => r.cls === 'family' && r.note.includes('rows=1554; complete=yes')));
assert.equal(p.environment.landlockAbi, 8);

// a number that does not add up fails the build
const rows = parseSummary('table,case,rabbit_verdict,linux_verdict,reached_linux,class,note\nadversary,x,denied,allowed,na,policy,\n');
assert.throws(() => verifyTotals(rows, parseTotals('table,class,count\nadversary,policy,2\nadversary,all,1\n')), /does not|says/);
// a changed header fails
assert.throws(() => parseSummary('table,case\n'), /header/);
console.log('policy-check: all assertions passed');
