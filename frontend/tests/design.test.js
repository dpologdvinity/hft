import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('durable design token values match the canonical runtime stylesheet', () => {
  const design = readFileSync(new URL('../../DESIGN.md', import.meta.url), 'utf8');
  const css = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');
  for (const match of design.matchAll(/^  ([\w-]+): "(#[0-9a-f]{6})"$/gm)) {
    assert.ok(css.includes(`--color-${match[1]}: ${match[2]};`), `Color ${match[1]} drifted`);
  }
  for (const [name, expected] of [
    ['panel', '10px'],
    ['control', '6px'],
  ]) {
    assert.ok(css.includes(`--radius-${name}: ${expected};`));
  }
});
