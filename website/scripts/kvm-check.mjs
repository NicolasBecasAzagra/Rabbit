// Checks of src/lib/kvm.ts. Run: node --experimental-strip-types scripts/kvm-check.mjs
import assert from 'node:assert/strict';
import { loadKvm, parseCsv, withUnit } from '../src/lib/kvm.ts';

assert.equal(withUnit(376.5, 'ns/op'), '376.5 ns');
assert.equal(withUnit(1309, 'ns/op'), '1.309 us');
assert.equal(withUnit(6.83e6, 'ns/MiB'), '6.83 ms');
assert.equal(withUnit(11250, 'ms'), '11.25 s');
assert.equal(withUnit(109.7, 'ms'), '109.7 ms');
assert.equal(withUnit(46.41, 'tok/s'), '46.41 tok/s');
assert.deepEqual(parseCsv('a,b\n1,2\n', ['a'], 'x'), [{ a: '1', b: '2' }]);
assert.throws(() => parseCsv('a,b\n', ['a'], 'x'), /no rows/);
assert.throws(() => parseCsv('a,b\n1,2\n', ['c'], 'x'), /no column "c"/);
assert.throws(() => parseCsv('a,b\n1,2,3\n', ['a'], 'x'), /fields/);

const rows = loadKvm();
console.log('kvm rows', rows.length);
const syscall = rows.find((r) => r.metric === 'System call');
assert.ok(syscall, 'the system call row is in the table');
assert.equal(syscall.ratioKvm, '1.17x');
assert.equal(syscall.rabbitKvm, '42 ns');
// Every row says where each figure was measured through its reading, and a row
// that reversed is never presented as one that held.
for (const row of rows) {
  assert.ok(['same side', 'reverses', 'tie under KVM', 'tie under one mode', 'unknown'].includes(row.reading), row.metric);
  if (row.reading === 'reverses') {
    const tcg = Number(row.ratioTcg.replace('x', ''));
    const kvm = Number(row.ratioKvm.replace('x', ''));
    assert.ok((tcg > 1) !== (kvm > 1), `${row.metric} is marked reversed but its ratios are ${row.ratioTcg} and ${row.ratioKvm}`);
  }
}
assert.ok(rows.some((r) => r.reading === 'reverses'), 'the losses of the KVM run are in the table');
console.log('kvm-check: all assertions passed');
