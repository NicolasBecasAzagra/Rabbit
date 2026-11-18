// The status and the next steps of the project, read at build time from the
// README of the repository, so the site cannot drift from it. The build fails
// with a clear message when a table changes shape or when the README says a
// milestone is done that the data of the site does not know.

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { milestones } from '../data/project.ts';

export type StatusRow = { id: string; delivers: string };
export type NextRow = { title: string; issues: { number: number; url: string }[]; why: string; deferred: boolean };

function fail(message: string): never {
  throw new Error(`[readme] ${message}`);
}

function locate(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  // The README of the repository, not the one of website/: it has the status table.
  const candidates = [resolve(here, '../../../README.md'), resolve(process.cwd(), '../README.md')];
  const found = candidates.find((p) => existsSync(p) && readFileSync(p, 'utf8').includes('\n## What comes next\n'));
  if (!found) fail(`README.md not found. Looked in:\n  ${candidates.join('\n  ')}`);
  return found;
}

/** Plain text of a markdown cell: no bold, no code ticks, links reduced to their text. */
export function plain(cell: string): string {
  return cell
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .trim();
}

export function tableOf(markdown: string, heading: string, header: string[]): string[][] {
  const start = markdown.indexOf(`\n## ${heading}\n`);
  if (start < 0) fail(`the README has no "## ${heading}" section`);
  const rest = markdown.slice(start + 1);
  const end = rest.indexOf('\n## ', 4);
  const section = end < 0 ? rest : rest.slice(0, end);
  const lines = section.split('\n').filter((l) => l.trim().startsWith('|'));
  if (lines.length < 3) fail(`the table of "## ${heading}" is missing`);
  const cells = (line: string) =>
    line
      .trim()
      .replace(/^\||\|$/g, '')
      .split(/(?<!\\)\|/)
      .map((c) => c.trim());
  const head = cells(lines[0]).map(plain);
  if (head.join('|') !== header.join('|')) fail(`"## ${heading}" header is [${head}], expected [${header}]`);
  return lines.slice(2).map((line) => {
    const row = cells(line);
    if (row.length !== header.length) fail(`a row of "## ${heading}" has ${row.length} cells: ${line}`);
    return row;
  });
}

export function parseReadme(markdown: string) {
  const status: StatusRow[] = tableOf(markdown, 'Status', ['Milestone', 'What it delivers', 'State']).map((row) => {
    if (plain(row[2]) !== 'done') fail(`status row ${row[0]} is "${row[2]}", the site only draws done rows here`);
    return { id: plain(row[0]), delivers: plain(row[1]) };
  });
  const ids = new Set<string>();
  for (const row of status) {
    if (ids.has(row.id)) fail(`milestone ${row.id} appears twice in the Status table`);
    ids.add(row.id);
  }
  const next: NextRow[] = tableOf(markdown, 'What comes next', ['Milestone', 'Issues', 'Why it is next']).map((row) => {
    const issues = [...row[1].matchAll(/\[#(\d+)\]\(([^)]+)\)/g)].map((m) => ({ number: Number(m[1]), url: m[2] }));
    const title = plain(row[0]);
    if (!issues.length) fail(`the next row "${title}" lists no issue`);
    return { title, issues, why: plain(row[2]), deferred: title === 'Deferred' };
  });
  if (!next.length || next[0].deferred) fail('the first row of "What comes next" must be a real milestone');
  return { status, next };
}

export function loadReadme() {
  const data = parseReadme(readFileSync(locate(), 'utf8'));
  const known = new Map(milestones.map((m) => [m.id, m]));
  for (const row of data.status) {
    const m = known.get(row.id);
    if (!m) fail(`the README says ${row.id} is done, but src/data/project.ts has no such milestone`);
    if (m.state !== 'done') fail(`the README says ${row.id} is done, but src/data/project.ts says ${m.state}`);
  }
  return data;
}
