import { useEffect, useRef, useState } from 'react';
import Icon from './Icon';
import MarketHistory from './MarketHistory';
import ResearchRuns from './ResearchRuns';
import TrainingRun from './TrainingRun';
import { currency, number, percent, sessionDate, sourceLabel, timestamp } from './format';
import { EmptyState, Panel, Status } from './primitives';
import { useDashboard } from './useDashboard';
import { TradingBots } from './TradingBots';

const views = {
  overview: {
    label: 'Overview',
    title: 'Research overview',
    description: 'Train on history. Validate on unseen sessions.',
  },
  research: {
    label: 'Research',
    title: 'Research',
    description: 'Inspect local runs and the evidence behind each result.',
  },
  paper: {
    label: 'Paper trading',
    title: 'Paper trading',
    description: 'Research eligibility and recorded paper-trading evidence.',
  },
};
function currentView() {
  const view = window.location.hash.slice(1);
  return Object.hasOwn(views, view) ? view : 'overview';
}

function Summary({ dataset, training }) {
  return (
    <section className="summary-band" aria-label="Dataset summary">
      <div>
        <span className="metric-label">Dataset</span>
        <strong>
          {number(dataset.total_sessions)} <span>sessions</span>
        </strong>
        <span className="metric-context">
          {[dataset.symbol, sourceLabel(dataset.source)].filter(Boolean).join(' · ') ||
            'No dataset loaded'}
        </span>
      </div>
      <div>
        <span className="metric-label">Development</span>
        <strong>
          {number(dataset.development_sessions)} <span>sessions</span>
        </strong>
        <span className="metric-context">
          {dataset.development_start_date && dataset.development_end_date
            ? `${sessionDate(dataset.development_start_date)} – ${sessionDate(dataset.development_end_date)}`
            : 'Training and validation'}
        </span>
      </div>
      <div>
        <span className="metric-label">Final test</span>
        <strong>
          {number(dataset.final_test_sessions)}{' '}
          <span>{training.held_out_evaluated ? 'evaluated' : 'reserved'}</span>
        </strong>
        <span className="metric-context">
          {training.held_out_evaluated ? 'Evaluation recorded' : 'Held out from development'}
        </span>
      </div>
      <div>
        <span className="metric-label">Starting capital</span>
        <strong>{currency(dataset.initial_cash)}</strong>
        <span className="metric-context">Simulated</span>
      </div>
    </section>
  );
}

function ResearchProtocol({ dataset, training }) {
  return (
    <Panel
      title="Evaluation protocol"
      subtitle="The final test stays separate from model development."
      headingId="protocol-heading"
    >
      <dl className="protocol-list">
        <div>
          <dt>Development history</dt>
          <dd>{number(dataset.development_sessions)} sessions for training and validation</dd>
        </div>
        <div>
          <dt>Reserved final test</dt>
          <dd>
            {number(dataset.final_test_sessions)} sessions ·{' '}
            {training.held_out_evaluated ? 'evaluation recorded' : 'not yet evaluated'}
          </dd>
        </div>
        <div>
          <dt>Training budget</dt>
          <dd>{number(training.steps_per_trial)} steps per trial</dd>
        </div>
        <div>
          <dt>Research eligibility</dt>
          <dd>
            {training.paper_eligible
              ? 'Recorded as eligible for paper trading'
              : 'Not yet established'}
          </dd>
        </div>
      </dl>
    </Panel>
  );
}

function PaperView({ paper, training }) {
  const value = paper || {};
  const metricLabels = {
    sessions: 'Recorded sessions',
    completed_sessions: 'Complete sessions',
    trades: 'Completed trades',
    initial_cash: 'Starting cash',
    equity: 'Recorded equity',
    net_profit: 'Net profit',
    max_drawdown: 'Maximum drawdown',
    expectancy: 'Expectancy per trade',
    source: 'Evidence source',
  };
  const cashMetrics = ['initial_cash', 'equity', 'net_profit', 'expectancy'];
  const sessions = Array.isArray(value.sessions) ? value.sessions : [];
  function displayMetric(key, metric) {
    if (metric === null || metric === undefined) return 'Not recorded';
    if (cashMetrics.includes(key)) return currency(metric, 2);
    if (key === 'max_drawdown') return Number.isFinite(metric) ? percent(metric) : 'Not recorded';
    return typeof metric === 'number'
      ? number(metric)
      : typeof metric === 'string'
        ? metric
        : 'Not recorded';
  }
  return (
    <div className="paper-layout">
      <TradingBots />
      <Panel
        title="Paper-trading eligibility"
        actions={<Status value={value.status} />}
        headingId="paper-heading"
      >
        <div className="paper-state">
          <span
            className={`paper-mark ${value.eligible ? 'paper-mark--eligible' : ''}`}
            aria-hidden="true"
          >
            <Icon name={value.eligible ? 'check' : 'paper'} />
          </span>
          <h3>{value.eligible ? 'Research gate passed' : 'Awaiting validated research'}</h3>
          <p>{value.message || 'No paper-trading eligibility report is available.'}</p>
        </div>
        <dl className="protocol-list">
          <div>
            <dt>Held-out evaluation</dt>
            <dd>{training.held_out_evaluated ? 'Recorded' : 'Not completed'}</dd>
          </div>
          <div>
            <dt>Research eligibility</dt>
            <dd>{value.eligible ? 'Eligible' : 'Not established'}</dd>
          </div>
          <div>
            <dt>Account activity</dt>
            <dd>This read-only view does not connect to or operate an account.</dd>
          </div>
        </dl>
      </Panel>
      <Panel
        title="Recorded results"
        subtitle="Only verified paper summaries appear here."
        headingId="paper-results-heading"
      >
        {value.metrics && typeof value.metrics === 'object' && Object.keys(value.metrics).length ? (
          <dl className="protocol-list">
            {Object.entries(value.metrics).map(([key, metric]) => (
              <div key={key}>
                <dt>{metricLabels[key] || key.replaceAll('_', ' ')}</dt>
                <dd>{displayMetric(key, metric)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <EmptyState title="No verified paper results">
            Profit, trades, and account balances are unavailable until real paper evidence is
            recorded.
          </EmptyState>
        )}
      </Panel>
      {sessions.length > 0 && (
        <Panel
          title="Recorded paper sessions"
          subtitle="Verified local journals. Incomplete sessions remain identified."
          className="paper-sessions-panel"
          headingId="paper-sessions-heading"
        >
          <div
            className="table-scroll chart-table"
            tabIndex="0"
            role="region"
            aria-label="Recorded paper sessions"
          >
            <table className="paper-sessions-table">
              <caption className="visually-hidden">Recorded paper session evidence</caption>
              <thead>
                <tr>
                  <th scope="col">Session</th>
                  <th scope="col">Coverage</th>
                  <th scope="col" className="numeric">
                    Closing cash
                  </th>
                  <th scope="col" className="numeric">
                    Position
                  </th>
                  <th scope="col">Source</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((session, index) => (
                  <tr key={`${session.date}-${index}`}>
                    <th scope="row">
                      {sessionDate(session.date, {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                      })}
                    </th>
                    <td>
                      {session.complete === true
                        ? 'Complete'
                        : session.complete === false
                          ? 'Incomplete'
                          : 'Not recorded'}
                    </td>
                    <td className="numeric">{currency(session.cash, 2)}</td>
                    <td className="numeric">{number(session.position)}</td>
                    <td>{session.source || 'Not recorded'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}

export default function App() {
  const [view, setView] = useState(currentView);
  const { data, loading, error, refresh } = useDashboard();
  const headingRef = useRef(null);
  const firstView = useRef(true);
  useEffect(() => {
    const change = () => {
      const next = window.location.hash.slice(1);
      if (Object.hasOwn(views, next)) setView(next);
    };
    window.addEventListener('hashchange', change);
    return () => window.removeEventListener('hashchange', change);
  }, []);
  useEffect(() => {
    document.title = `${views[view].title} — HFT`;
    if (firstView.current) firstView.current = false;
    else headingRef.current?.focus();
  }, [view]);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="HFT research overview">
          <svg viewBox="0 0 36 42" aria-hidden="true">
            <path d="M2 18h7v15l-7 4zM14 9l7 4v25l-7 4zM26 0l7 5v29l-7-4z" fill="currentColor" />
          </svg>
          <span>HFT</span>
        </a>
        <nav className="main-nav" aria-label="Main navigation">
          {Object.entries(views).map(([key, item]) => (
            <a href={`#${key}`} key={key} aria-current={view === key ? 'page' : undefined}>
              <Icon name={key} />
              <span>{item.label}</span>
            </a>
          ))}
        </nav>
        <div className="workspace-label">
          <Icon name="folder" />
          <span>Local workspace</span>
          <span className="read-only-label">Read only</span>
        </div>
      </aside>
      <main id="main-content" className="main-content">
        <header className="page-header">
          <div>
            <h1 tabIndex="-1" ref={headingRef}>
              {views[view].title}
            </h1>
            <p>{views[view].description}</p>
          </div>
          <div className="header-actions">
            <button
              className="primary-button refresh-button"
              type="button"
              onClick={refresh}
              disabled={loading}
              aria-busy={loading}
            >
              <Icon name="refresh" className={loading ? 'refreshing' : ''} />
              <span>{loading ? 'Loading' : 'Refresh'}</span>
            </button>
            <div className={`freshness ${error ? 'freshness--stale' : ''}`} role="status">
              <span className="status-dot" aria-hidden="true" />
              <span>
                {loading
                  ? data
                    ? 'Refreshing snapshot'
                    : 'Loading local snapshot'
                  : error
                    ? data
                      ? 'Stale snapshot'
                      : 'Unavailable'
                    : data?.updated_at
                      ? `Read ${timestamp(data.updated_at)}`
                      : 'Snapshot time unavailable'}
              </span>
            </div>
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            <strong>{data ? 'Refresh unavailable.' : 'Dashboard unavailable.'}</strong>
            <span>
              {error}{' '}
              {data
                ? `Showing the snapshot read ${timestamp(data.updated_at)}.`
                : 'Use Refresh to try again.'}
            </span>
          </div>
        )}
        {!data ? (
          <section className="initial-state" aria-live="polite" aria-busy={loading}>
            <div className={loading ? 'loading-indicator' : 'offline-indicator'} aria-hidden="true">
              {loading ? <Icon name="refresh" /> : <Icon name="research" />}
            </div>
            <h2>{loading ? 'Reading your local research' : 'No local snapshot available'}</h2>
            <p>
              {loading
                ? 'Loading dataset history and recorded reports.'
                : 'The dashboard needs its local API server to display real data.'}
            </p>
          </section>
        ) : (
          <>
            <Summary dataset={data.dataset} training={data.training} />
            {view === 'overview' && (
              <>
                <div className="overview-grid">
                  <MarketHistory data={data} />
                  <TrainingRun training={data.training} />
                </div>
                <ResearchRuns runs={data.runs} />
              </>
            )}
            {view === 'research' && (
              <>
                <div className="research-grid">
                  <ResearchProtocol dataset={data.dataset} training={data.training} />
                  <TrainingRun training={data.training} />
                </div>
                <ResearchRuns runs={data.runs} />
              </>
            )}
            {view === 'paper' && <PaperView paper={data.paper} training={data.training} />}
            {[
              ...(Array.isArray(data.issues) ? data.issues : []),
              ...(Array.isArray(data.warnings) ? data.warnings : []),
            ].length > 0 && (
              <section className="notice-panel" aria-label="Research notes">
                <h2>Research notes</h2>
                <ul>
                  {[
                    ...new Set([
                      ...(Array.isArray(data.issues) ? data.issues : []),
                      ...(Array.isArray(data.warnings) ? data.warnings : []),
                    ]),
                  ].map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </section>
            )}
            <footer className="page-footer">
              {view === 'paper'
                ? 'Paper eligibility is a research gate. Real paper evidence is required before any live consideration.'
                : 'Paper eligibility requires a completed evaluation.'}
              <span>Read-only local dashboard</span>
            </footer>
          </>
        )}
      </main>
    </div>
  );
}
