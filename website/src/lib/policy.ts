// The tool policy dataset, read at build time from bench/results/policy/.
// No number of the policy section is typed in a component: every count is
// computed here from summary.csv, cross-checked against totals.csv, and the
// build fails with a clear message when a file is missing or a figure does
// not add up.

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export type PolicyRow = {
  table: 'adversary' | 'compromised';
  case: string;
  rabbit: string;
  linux: string;
  reachedLinux: 'yes' | 'no' | 'na';
  cls: string;
  note: string;
};

export type PolicyEnvironment = {
  linuxRelease: string;
  landlockAbi: number;
  udpPortRules: string;
  ciRunId: number;
  commit: string;
};

function fail(message: string): never {
  throw new Error(`[policy] ${message}`);
}

function locateDataset(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const candidates = [
    resolve(here, '../../../bench/results/policy'),
    resolve(process.cwd(), '../bench/results/policy'),
    resolve(process.cwd(), 'bench/results/policy'),
  ];
  const found = candidates.find((dir) => existsSync(resolve(dir, 'summary.csv')));
  if (!found) {
    fail(`bench/results/policy/summary.csv not found. Looked in:\n  ${candidates.join('\n  ')}`);
  }
  return found;
}

function readFile(dir: string, name: string): string {
  const path = resolve(dir, name);
  if (!existsSync(path)) fail(`missing dataset file ${path}`);
  return readFileSync(path, 'utf8');
}

/** RFC 4180 subset: comma separated, double quoted fields, "" escapes a quote. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') {
        quoted = false;
      } else {
        field += c;
      }
    } else if (c === '"') {
      quoted = true;
    } else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field);
      field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else {
      field += c;
    }
  }
  if (quoted) fail('unterminated quoted field in a CSV file');
  if (field !== '' || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

const summaryHeader = ['table', 'case', 'rabbit_verdict', 'linux_verdict', 'reached_linux', 'class', 'note'];

export function parseSummary(text: string): PolicyRow[] {
  const [header, ...body] = parseCsv(text);
  if (!header || header.join(',') !== summaryHeader.join(',')) {
    fail(`summary.csv header is [${header?.join(',')}], expected [${summaryHeader.join(',')}]`);
  }
  return body.map((cells, index) => {
    if (cells.length !== summaryHeader.length) {
      fail(`summary.csv line ${index + 2} has ${cells.length} fields, expected ${summaryHeader.length}`);
    }
    const [table, name, rabbit, linux, reached, cls, note] = cells;
    if (table !== 'adversary' && table !== 'compromised') fail(`unknown table "${table}" on line ${index + 2}`);
    if (reached !== 'yes' && reached !== 'no' && reached !== 'na') {
      fail(`unknown reached_linux "${reached}" on line ${index + 2}`);
    }
    return { table, case: name, rabbit, linux, reachedLinux: reached, cls, note };
  });
}

export function parseTotals(text: string): Map<string, number> {
  const [header, ...body] = parseCsv(text);
  if (!header || header.join(',') !== 'table,class,count') fail('totals.csv header is not table,class,count');
  const totals = new Map<string, number>();
  for (const [table, cls, count] of body) {
    const n = Number(count);
    if (!Number.isInteger(n)) fail(`totals.csv: "${count}" is not a count (${table}/${cls})`);
    totals.set(`${table}/${cls}`, n);
  }
  return totals;
}

/** The enumerated family and the existence oracle are not single cases: totals.csv does not count them. */
export const isCase = (row: PolicyRow) => row.cls !== 'family' && !row.case.startsWith('existence:');

/** Every class counted from the rows, compared with totals.csv; throws on any difference. */
export function verifyTotals(rows: PolicyRow[], totals: Map<string, number>): Map<string, number> {
  const counted = new Map<string, number>();
  for (const row of rows.filter(isCase)) {
    for (const key of [`${row.table}/${row.cls}`, `${row.table}/all`]) {
      counted.set(key, (counted.get(key) ?? 0) + 1);
    }
  }
  for (const [key, expected] of totals) {
    const actual = counted.get(key) ?? 0;
    if (actual !== expected) fail(`totals.csv says ${key} = ${expected}, summary.csv has ${actual} rows`);
  }
  for (const key of counted.keys()) {
    if (!totals.has(key)) fail(`summary.csv has rows of ${key} that totals.csv does not list`);
  }
  return counted;
}

function loadEnvironment(json: unknown): PolicyEnvironment {
  const j = json as any;
  const linux = j?.environment?.linux_runner;
  if (!linux || typeof linux.uname_r !== 'string' || !Number.isInteger(linux.landlock_abi)) {
    fail('summary.json has no environment.linux_runner.uname_r and landlock_abi');
  }
  return {
    linuxRelease: linux.uname_r,
    landlockAbi: linux.landlock_abi,
    udpPortRules: String(linux.udp_port_rules ?? ''),
    ciRunId: j.source?.ci_run_id,
    commit: String(j.source?.commit ?? '').slice(0, 7),
  };
}

export type PolicyData = ReturnType<typeof loadPolicy>;

export function loadPolicy() {
  const dir = locateDataset();
  const rows = parseSummary(readFile(dir, 'summary.csv'));
  const totals = parseTotals(readFile(dir, 'totals.csv'));
  verifyTotals(rows, totals);
  let json: unknown;
  try {
    json = JSON.parse(readFile(dir, 'summary.json'));
  } catch (error) {
    fail(`summary.json is not valid JSON: ${(error as Error).message}`);
  }
  const environment = loadEnvironment(json);

  const cases = rows.filter(isCase);
  const headline = cases.filter((row) => row.cls === 'policy');
  const compromised = cases.filter((row) => row.table === 'compromised');

  const counts = {
    /** cases Rabbit refused with its policy while Linux did not */
    refusedByPolicy: headline.filter((row) => row.rabbit === 'denied').length,
    /** of those, the ones Linux answered with `allowed` */
    linuxLetThrough: headline.filter((row) => row.linux === 'allowed').length,
    /** compromised raw calls Rabbit answers with ENOSYS */
    rabbitUnavailable: compromised.filter((row) => row.rabbit === 'unavailable').length,
    compromisedCalls: compromised.length,
    /** every case the two adversaries ran, and how they split by class */
    totalCases: cases.length,
    sameOnBothSides: cases.filter((row) => row.cls === 'same').length,
    differByPolicy: headline.length,
    differForOtherReasons: cases.filter((row) => row.cls !== 'same' && row.cls !== 'policy').length,
  };
  if (counts.sameOnBothSides + counts.differByPolicy + counts.differForOtherReasons !== counts.totalCases) {
    fail('the classes of summary.csv do not add up to the number of cases');
  }
  if (counts.refusedByPolicy !== headline.length) {
    fail('a policy row is not a Rabbit refusal; the headline ledger assumes it is');
  }
  return { rows, cases, headline, counts, environment, totals };
}
