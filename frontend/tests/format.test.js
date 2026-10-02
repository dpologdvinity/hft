import test from 'node:test';
import assert from 'node:assert/strict';
import {
  basisPoints,
  currency,
  marketPoints,
  percent,
  sessionDate,
  trialProgress,
} from '../src/format.js';

test('session dates stay on the same exchange date rather than shifting at midnight UTC', () => {
  assert.equal(
    sessionDate('2026-06-04', { month: 'short', day: 'numeric', year: 'numeric' }),
    'Jun 4, 2026',
  );
  assert.equal(sessionDate('not-a-date'), '—');
});

test('missing numerical evidence is unavailable rather than zero', () => {
  assert.equal(currency(null), '—');
  assert.deepEqual(trialProgress({}), { total: null, completed: null, percent: null });
  assert.deepEqual(trialProgress({ completed_trials: 0, total_trials: 18 }), {
    total: 18,
    completed: 0,
    percent: 0,
  });
});

test('market month range is relative to the last actual observation', () => {
  const history = [
    { date: '2026-07-14', close: 302 },
    { date: '2026-06-04', close: 301 },
    { date: '2026-06-16', close: 303 },
    { date: '2026-06-17', close: null },
  ];
  assert.deepEqual(
    marketPoints(history, 'month').map((point) => point.date),
    ['2026-06-16', '2026-07-14'],
  );
  assert.equal(marketPoints(history, 'all').length, 3);
  assert.deepEqual(marketPoints(undefined, 'all'), []);
});

test('financial ratios and daily-return intervals preserve their real units', () => {
  assert.equal(percent(0.05), '5%');
  assert.equal(basisPoints(0.001), '10 bps');
  assert.equal(basisPoints(-0.0005), '-5 bps');
  assert.equal(percent(null), '—');
});
