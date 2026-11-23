// The table of the run under KVM, read at build time from the files that
// bench/kvm/build_table.py derived, so the page cannot say anything the data does
// not. The build fails with a clear message when a file is missing or its columns
// change, the same way src/lib/readme.ts does for the README.

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export type KvmRow = {
  metric: string;
  rabbitTcg: string;
  referenceTcg: string;
  ratioTcg: string;
  rabbitKvm: string;
  referenceKvm: string;
  ratioKvm: string;
  reading: 'same side' | 'reverses' | 'tie under KVM' | 'tie under one mode' | 'unknown';
};

const RATIO_COLUMNS = ['dataset', 'workload', 'kind', 'reference', 'ratio_tcg', 'ratio_kvm', 'direction'];
const ROW_COLUMNS = ['dataset', 'accel', 'runs', 'kernel', 'workload', 'unit', 'median'];

/** The same names the Markdown table of the repository uses. */
const LABELS: Record<string, string> = {
  syscall: 'System call',
  mmap: 'Map and touch memory, per MiB',
  weights: 'Fault in memory, 4 KiB pages, per MiB',
  'weights-tail': 'Slowest single fault, 4 KiB',
  'weights-huge': 'Fault in memory, 2 MiB pages, per MiB',
  'weights-huge-tail': 'Slowest single fault, 2 MiB',
  futex: 'Futex round trip',
  threads: 'Four threads, spawn to join',
  'infer-ttft': 'Time to the first token',
  'infer-tps': 'Tokens per second',
  'agent-seconds': 'Seconds of an agent loop',
};

function fail(message: string): never {
  throw new Error(`[kvm] ${message}`);
}

function directory(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const candidates = [
    resolve(here, '../../../bench/results/kvm/table'),
    resolve(process.cwd(), '../bench/results/kvm/table'),
  ];
  const found = candidates.find((p) => existsSync(resolve(p, 'ratios.csv')));
  if (!found) fail(`bench/results/kvm/table/ratios.csv not found. Looked in:\n  ${candidates.join('\n  ')}`);
  return found;
}

/** A CSV of the rig: no quoted fields, a header, one record per line. */
export function parseCsv(text: string, expected: string[], name: string): Record<string, string>[] {
  const lines = text.split('\n').filter((line) => line.length > 0);
  if (lines.length < 2) fail(`${name} has no rows`);
  const header = lines[0].split(',');
  for (const column of expected) {
    if (!header.includes(column)) fail(`${name} has no column "${column}"; its header is ${header.join(', ')}`);
  }
  return lines.slice(1).map((line) => {
    const cells = line.split(',');
    if (cells.length !== header.length) fail(`${name}: a row has ${cells.length} fields and the header ${header.length}: ${line}`);
    return Object.fromEntries(header.map((h, i) => [h, cells[i]]));
  });
}

/** A figure the way a person reads it, with the unit that keeps it between 1 and 1000. */
export function withUnit(value: number, unit: string): string {
  const four = (x: number) => Number(x.toPrecision(4)).toString();
  if (unit.startsWith('ns')) {
    if (value < 1e3) return `${four(value)} ns`;
    if (value < 1e6) return `${four(value / 1e3)} us`;
    if (value < 1e9) return `${four(value / 1e6)} ms`;
    return `${four(value / 1e9)} s`;
  }
  if (unit === 'ms' && value >= 1000) return `${four(value / 1000)} s`;
  return `${four(value)} ${unit}`;
}

function ratioText(text: string): string {
  return text === '' ? '-' : `${Number(text).toFixed(2)}x`;
}

export function loadKvm(): KvmRow[] {
  const dir = directory();
  const ratios = parseCsv(readFileSync(resolve(dir, 'ratios.csv'), 'utf8'), RATIO_COLUMNS, 'ratios.csv');
  const rows = parseCsv(readFileSync(resolve(dir, 'rows.csv'), 'utf8'), ROW_COLUMNS, 'rows.csv');
  const median = (dataset: string, accel: string, kernel: string, workload: string): string => {
    const row = rows.find((r) => r.dataset === dataset && r.accel === accel && r.kernel === kernel && r.workload === workload);
    const value = row ? Number(row.median) : NaN;
    return row && Number.isFinite(value) ? withUnit(value, row.unit) : '-';
  };
  const out: KvmRow[] = [];
  for (const r of ratios) {
    if (r.reference !== 'mainline-nomitig' || r.kind !== 'comparison' || !(r.workload in LABELS)) continue;
    out.push({
      metric: LABELS[r.workload],
      rabbitTcg: median(r.dataset, 'tcg', 'rabbit', r.workload),
      referenceTcg: median(r.dataset, 'tcg', 'mainline-nomitig', r.workload),
      ratioTcg: ratioText(r.ratio_tcg),
      rabbitKvm: median(r.dataset, 'kvm', 'rabbit', r.workload),
      referenceKvm: median(r.dataset, 'kvm', 'mainline-nomitig', r.workload),
      ratioKvm: ratioText(r.ratio_kvm),
      reading: r.direction as KvmRow['reading'],
    });
  }
  if (out.length === 0) fail('no comparison rows against mainline-nomitig in ratios.csv');
  return out;
}
