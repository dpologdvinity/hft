const locale = 'en-US';
const zone = 'America/New_York';

export const number = (value) =>
  Number.isFinite(value) ? new Intl.NumberFormat(locale).format(value) : '—';

export const currency = (value, digits = 0) =>
  Number.isFinite(value)
    ? new Intl.NumberFormat(locale, {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      }).format(value)
    : '—';

export const sourceLabel = (value) => (value === 'iex' ? 'IEX' : value);

export const percent = (value) =>
  Number.isFinite(value)
    ? new Intl.NumberFormat(locale, { style: 'percent', maximumFractionDigits: 2 }).format(value)
    : '—';

export const basisPoints = (value) =>
  Number.isFinite(value)
    ? `${new Intl.NumberFormat(locale, { maximumFractionDigits: 2 }).format(value * 10000)} bps`
    : '—';

// Noon UTC keeps an exchange-session date on the same date in New York.
export function sessionDate(value, options = { month: 'short', day: 'numeric' }) {
  if (!value) return '—';
  const date = new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T12:00:00Z` : value);
  return Number.isFinite(date.getTime())
    ? new Intl.DateTimeFormat(locale, { ...options, timeZone: zone }).format(date)
    : '—';
}

export function timestamp(value) {
  return sessionDate(value, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    timeZoneName: 'short',
  });
}

export const statusLabels = {
  idle: 'Not started',
  running: 'Running',
  interrupted: 'Interrupted',
  'incomplete-search': 'Incomplete search',
  'failed-edge': 'Edge not established',
  'paper-eligible': 'Paper eligible',
  'insufficient-data': 'Insufficient data',
  error: 'Error',
  failed: 'Failed',
  completed: 'Completed',
  'awaiting-validation': 'Awaiting validation',
  unproven: 'Unproven',
  available: 'Available',
  observed: 'Paper results recorded',
};

export function statusLabel(value) {
  return (
    statusLabels[value] ||
    (typeof value === 'string' ? value.replaceAll('-', ' ') : 'Not available')
  );
}

export function statusTone(value) {
  if (['completed', 'paper-eligible', 'available'].includes(value)) return 'success';
  if (['error', 'failed', 'failed-edge'].includes(value)) return 'danger';
  if (['interrupted', 'incomplete-search', 'insufficient-data', 'unproven'].includes(value))
    return 'warning';
  return 'neutral';
}

export function marketPoints(history, range) {
  const valid = (Array.isArray(history) ? history : [])
    .filter(
      (point) =>
        typeof point.date === 'string' &&
        /^\d{4}-\d{2}-\d{2}$/.test(point.date) &&
        Number.isFinite(new Date(`${point.date}T12:00:00Z`).getTime()) &&
        Number.isFinite(point.close) &&
        point.close > 0,
    )
    .sort((a, b) => a.date.localeCompare(b.date));
  if (range !== 'month' || !valid.length) return valid;
  const cutoff = new Date(`${valid.at(-1).date}T12:00:00Z`);
  cutoff.setUTCMonth(cutoff.getUTCMonth() - 1);
  const date = cutoff.toISOString().slice(0, 10);
  return valid.filter((point) => point.date >= date);
}

export function trialProgress(training = {}) {
  const total =
    Number.isFinite(training.total_trials) && training.total_trials > 0
      ? training.total_trials
      : null;
  const completed =
    Number.isFinite(training.completed_trials) && training.completed_trials >= 0
      ? training.completed_trials
      : null;
  return {
    total,
    completed,
    percent: total !== null && completed !== null ? Math.min(100, (completed / total) * 100) : null,
  };
}
