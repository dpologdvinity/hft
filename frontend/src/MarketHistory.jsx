import { useEffect, useState } from 'react';
import { currency, marketPoints, sessionDate, sourceLabel } from './format';
import { EmptyState, Panel } from './primitives';

function PriceChart({ points }) {
  const [compact, setCompact] = useState(() => window.matchMedia('(max-width: 640px)').matches);
  useEffect(() => {
    const media = window.matchMedia('(max-width: 640px)');
    const update = () => setCompact(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  const width = compact ? 360 : 680,
    height = compact ? 226 : 286;
  const pad = { left: compact ? 48 : 54, right: 14, top: 16, bottom: compact ? 32 : 38 };
  const plotWidth = width - pad.left - pad.right,
    plotHeight = height - pad.top - pad.bottom;
  const prices = points.map((point) => point.close);
  const min = Math.min(...prices),
    max = Math.max(...prices);
  const margin = Math.max((max - min) * 0.17, max * 0.004);
  const low = min - margin,
    high = max + margin;
  const first = new Date(`${points[0].date}T12:00:00Z`).getTime();
  const last = new Date(`${points.at(-1).date}T12:00:00Z`).getTime();
  const x = (date) =>
    pad.left +
    (last === first
      ? plotWidth / 2
      : ((new Date(`${date}T12:00:00Z`).getTime() - first) / (last - first)) * plotWidth);
  const y = (price) => pad.top + ((high - price) / (high - low)) * plotHeight;
  const line = points
    .map(
      (point, index) =>
        `${index ? 'L' : 'M'}${x(point.date).toFixed(2)},${y(point.close).toFixed(2)}`,
    )
    .join(' ');
  const tickCount = Math.min(compact ? 4 : 5, points.length);
  const tickIndices = [
    ...new Set(
      Array.from({ length: tickCount }, (_, i) =>
        Math.round((i * (points.length - 1)) / (tickCount - 1 || 1)),
      ),
    ),
  ];

  return (
    <svg
      className="price-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-labelledby="chart-title chart-description"
    >
      <title id="chart-title">Historical daily last reported trade prices</title>
      <desc id="chart-description">
        {points.length} development sessions, from{' '}
        {sessionDate(points[0].date, { month: 'long', day: 'numeric', year: 'numeric' })} to{' '}
        {sessionDate(points.at(-1).date, { month: 'long', day: 'numeric', year: 'numeric' })}.
        Prices in US dollars. A data table is available below.
      </desc>
      {Array.from({ length: 5 }, (_, i) => {
        const value = low + ((high - low) * i) / 4,
          position = y(value);
        return (
          <g key={i}>
            <line
              className="chart-grid"
              x1={pad.left}
              x2={width - pad.right}
              y1={position}
              y2={position}
            />
            <text className="chart-label" x={pad.left - 10} y={position + 4} textAnchor="end">
              {currency(value, 0)}
            </text>
          </g>
        );
      })}
      {tickIndices.map((index) => (
        <g key={points[index].date}>
          <line
            className="chart-grid"
            x1={x(points[index].date)}
            x2={x(points[index].date)}
            y1={pad.top}
            y2={height - pad.bottom}
          />
          <text
            className="chart-label"
            x={x(points[index].date)}
            y={height - 12}
            textAnchor={index === 0 ? 'start' : index === points.length - 1 ? 'end' : 'middle'}
          >
            {sessionDate(points[index].date)}
          </text>
        </g>
      ))}
      <path className="chart-line" d={line} />
      {points.length === 1 && (
        <circle className="chart-point" cx={x(points[0].date)} cy={y(points[0].close)} r="4" />
      )}
    </svg>
  );
}

export default function MarketHistory({ data }) {
  const [range, setRange] = useState(() =>
    new URLSearchParams(window.location.search).get('range') === 'month' ? 'month' : 'all',
  );
  const [showTable, setShowTable] = useState(false);
  const points = marketPoints(data.market_history, range);
  function selectRange(value) {
    setRange(value);
    const url = new URL(window.location.href);
    if (value === 'month') url.searchParams.set('range', 'month');
    else url.searchParams.delete('range');
    window.history.replaceState(null, '', url);
  }
  const actions = (
    <div className="segmented" aria-label="Market history range">
      <button type="button" aria-pressed={range === 'all'} onClick={() => selectRange('all')}>
        All
      </button>
      <button type="button" aria-pressed={range === 'month'} onClick={() => selectRange('month')}>
        1 month
      </button>
    </div>
  );
  return (
    <Panel
      title="Market history"
      subtitle="Last reported trade · development sessions only"
      actions={actions}
      className="market-panel"
      headingId="market-heading"
    >
      {points.length ? (
        <>
          <PriceChart points={points} />
          <div className="chart-footer">
            <span>
              {data.dataset.symbol || 'Stock'} ·{' '}
              {sourceLabel(data.dataset.source) || 'Local history'} ·{' '}
              {sessionDate(points[0].date, { year: 'numeric' })}
            </span>
            <button
              type="button"
              className="text-button"
              aria-expanded={showTable}
              aria-controls={showTable ? 'market-data-table' : undefined}
              onClick={() => setShowTable(!showTable)}
            >
              {showTable ? 'Hide data table' : 'View data table'}
            </button>
          </div>
          {showTable && (
            <div
              className="table-scroll chart-table"
              id="market-data-table"
              tabIndex="0"
              role="region"
              aria-label="Development market prices"
            >
              <table>
                <caption className="visually-hidden">
                  Development sessions and daily last reported trade price; not strategy returns
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Session</th>
                    <th scope="col" className="numeric">
                      Last trade price
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {points.map((point) => (
                    <tr key={point.date}>
                      <td>
                        {sessionDate(point.date, {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric',
                        })}
                      </td>
                      <td className="numeric">{currency(point.close, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      ) : (
        <EmptyState title="No development prices available">
          Real historical prices will appear here when the local dataset is available.
        </EmptyState>
      )}
    </Panel>
  );
}
