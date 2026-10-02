import { Fragment, useState } from 'react';
import Icon from './Icon';
import { basisPoints, currency, number, percent } from './format';
import { EmptyState, Panel, Status } from './primitives';

function readableKey(key) {
  return key.replaceAll('_', ' ').replace(/\b\w/, (value) => value.toUpperCase());
}

function DetailValue({ value, field }) {
  if (value === null || value === undefined) return <>Not recorded</>;
  if (typeof value === 'boolean') return <>{value ? 'Yes' : 'No'}</>;
  if (typeof value === 'number')
    return <>{field === 'initial_cash' ? currency(value, 2) : number(value)}</>;
  if (typeof value === 'string') return <>{value}</>;
  if (Array.isArray(value))
    return value.length ? (
      <ul className="detail-list">
        {value.map((item, i) => (
          <li key={i}>
            <DetailValue value={item} />
          </li>
        ))}
      </ul>
    ) : (
      <>None recorded</>
    );
  return (
    <dl className="report-details report-details--nested">
      {Object.entries(value).map(([key, item]) => (
        <div key={key}>
          <dt>{readableKey(key)}</dt>
          <dd>
            <DetailValue value={item} field={key} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

function EvidenceMetrics({ values, labels, formatters = {} }) {
  return (
    <dl className="report-details">
      {Object.entries(labels)
        .filter(([key]) => Object.hasOwn(values, key))
        .map(([key, label]) => (
          <div key={key}>
            <dt>{label}</dt>
            <dd>
              {values[key] === null || values[key] === undefined
                ? 'Unavailable'
                : formatters[key]
                  ? formatters[key](values[key])
                  : typeof values[key] === 'number'
                    ? number(values[key])
                    : String(values[key])}
            </dd>
          </div>
        ))}
    </dl>
  );
}

function RunReport({ details }) {
  const { result, edge, decision_coverage: coverage, diagnostics, ...config } = details;
  return (
    <div className="report-sections">
      {Object.keys(config).length > 0 && (
        <section aria-label="Run configuration and reasons">
          <DetailValue value={config} />
        </section>
      )}
      {result && typeof result === 'object' && Object.keys(result).length > 0 && (
        <section className="report-evidence">
          <h4>Final-test results</h4>
          <p>
            Recorded held-out evaluation. Market prices are separate from these strategy results.
          </p>
          <EvidenceMetrics
            values={result}
            labels={{
              net_profit: 'Net profit',
              return: 'Strategy return',
              max_drawdown: 'Maximum drawdown',
              trades: 'Completed trades',
              expectancy: 'Expectancy per trade',
              profit_factor: 'Profit factor',
              sessions: 'Evaluated sessions',
            }}
            formatters={{
              net_profit: (value) => currency(value, 2),
              return: percent,
              max_drawdown: percent,
              expectancy: (value) => currency(value, 2),
            }}
          />
        </section>
      )}
      {edge && typeof edge === 'object' && Object.keys(edge).length > 0 && (
        <section className="report-evidence">
          <h4>Daily return advantage</h4>
          <p>Paired interval in basis points. 1 bp = 0.01 percentage points.</p>
          <EvidenceMetrics
            values={edge}
            labels={{
              primary_control: 'Primary control',
              lower: 'Interval lower bound',
              upper: 'Interval upper bound',
              positive: 'Positive advantage established',
            }}
            formatters={{
              primary_control: readableKey,
              lower: basisPoints,
              upper: basisPoints,
              positive: (value) =>
                value === true ? 'Yes' : value === false ? 'No' : 'Unavailable',
            }}
          />
        </section>
      )}
      {coverage && typeof coverage === 'object' && Object.keys(coverage).length > 0 && (
        <section className="report-evidence">
          <h4>Decision coverage</h4>
          <p>Recorded processing diagnostics from development data.</p>
          <EvidenceMetrics
            values={coverage}
            labels={{
              eligible_decisions: 'Eligible decisions',
              ineligible_decisions: 'Ineligible decisions',
              feed_gaps: 'Feed gaps',
              live_comparable: 'Comparable to live processing',
            }}
            formatters={{
              live_comparable: (value) =>
                value === true ? 'Yes' : value === false ? 'No' : 'Unavailable',
            }}
          />
        </section>
      )}
      {diagnostics && typeof diagnostics === 'object' && Object.keys(diagnostics).length > 0 && (
        <section className="report-evidence">
          <h4>Dataset diagnostics</h4>
          <DetailValue value={diagnostics} />
        </section>
      )}
    </div>
  );
}

export default function ResearchRuns({ runs }) {
  const records = Array.isArray(runs) ? runs : [];
  const [open, setOpen] = useState(null);
  const [page, setPage] = useState(0);
  const pageSize = 8,
    pageCount = Math.max(1, Math.ceil(records.length / pageSize));
  const safePage = Math.min(page, pageCount - 1);
  const visible = records.slice(safePage * pageSize, (safePage + 1) * pageSize);
  return (
    <Panel title="Research runs" className="research-panel" headingId="runs-heading">
      {records.length ? (
        <>
          <div
            className="table-scroll"
            tabIndex="0"
            role="region"
            aria-label="Research runs and reports"
          >
            <table className="runs-table">
              <caption className="visually-hidden">
                Local research reports. Select View report to inspect a run.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Data</th>
                  <th scope="col">Steps</th>
                  <th scope="col">Status</th>
                  <th scope="col">
                    <span className="visually-hidden">Report</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {visible.map((run, index) => {
                  const id = run.id || `run-${safePage * pageSize + index}`;
                  const expanded = open === id;
                  const detailId = `report-${safePage * pageSize + index}`;
                  return (
                    <Fragment key={id}>
                      <tr>
                        <th scope="row">{run.name || 'Unnamed run'}</th>
                        <td>{run.data || 'Not recorded'}</td>
                        <td>
                          {typeof run.steps === 'number'
                            ? `${number(run.steps)}${Number.isFinite(run.details?.timesteps) ? ' per trial' : ''}`
                            : run.steps || 'Not recorded'}
                        </td>
                        <td>
                          <Status value={run.status} />
                        </td>
                        <td className="report-action">
                          <button
                            className="text-button report-button"
                            type="button"
                            aria-expanded={expanded}
                            aria-controls={expanded ? detailId : undefined}
                            aria-label={`${expanded ? 'Hide' : 'View'} report for ${run.name || 'research run'}`}
                            onClick={() => setOpen(expanded ? null : id)}
                          >
                            {expanded ? 'Hide' : 'View'}
                            <Icon name="chevron" className={expanded ? 'chevron--open' : ''} />
                          </button>
                        </td>
                      </tr>
                      {expanded && (
                        <tr className="detail-row" id={detailId}>
                          <td colSpan="5">
                            <div className="report-body">
                              <h3>{run.name || 'Run'} report</h3>
                              {run.details && Object.keys(run.details).length ? (
                                <RunReport details={run.details} />
                              ) : (
                                <p>No additional report details are recorded.</p>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="table-hint">Scroll to view all columns.</p>
          {pageCount > 1 && (
            <div className="pagination">
              <span>
                {number(safePage * pageSize + 1)}–
                {number(Math.min((safePage + 1) * pageSize, records.length))} of{' '}
                {number(records.length)} runs
              </span>
              <nav aria-label="Research run pages">
                <button
                  type="button"
                  className="outline-button"
                  disabled={safePage === 0}
                  onClick={() => {
                    setPage(safePage - 1);
                    setOpen(null);
                  }}
                >
                  Previous
                </button>
                <button
                  type="button"
                  className="outline-button"
                  disabled={safePage === pageCount - 1}
                  onClick={() => {
                    setPage(safePage + 1);
                    setOpen(null);
                  }}
                >
                  Next
                </button>
              </nav>
            </div>
          )}
        </>
      ) : (
        <EmptyState compact title="No research reports yet">
          Completed or interrupted local runs will appear here.
        </EmptyState>
      )}
    </Panel>
  );
}
