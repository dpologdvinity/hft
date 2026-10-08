import { useEffect, useState } from 'react';
import { currency } from './format';
import { EmptyState, Panel } from './primitives';

const REFRESH_MS = 10000;

function useTrading() {
  const [state, setState] = useState({ runs: null, error: null });
  useEffect(() => {
    let ignore = false;
    async function load() {
      try {
        const response = await fetch('/api/trading', {
          cache: 'no-store',
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) throw new Error(`The local dashboard returned ${response.status}.`);
        const data = await response.json();
        if (!data || !Array.isArray(data.runs)) throw new Error('Unreadable trading response.');
        if (!ignore) setState({ runs: data.runs, error: null });
      } catch (error) {
        if (!ignore)
          setState((previous) => ({
            ...previous,
            error:
              error instanceof TypeError
                ? 'The local dashboard is unavailable. Check that its server is running.'
                : error.message,
          }));
      }
    }
    load();
    const interval = window.setInterval(load, REFRESH_MS);
    return () => {
      ignore = true;
      window.clearInterval(interval);
    };
  }, []);
  return state;
}

function money(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? currency(amount, 2) : 'Not recorded';
}

export function TradingBots() {
  const { runs, error } = useTrading();
  return (
    <Panel
      title="Paper trading bots"
      subtitle="Simulated-money runs started with hft trade --paper. Updates every 10 seconds."
      className="paper-sessions-panel"
      headingId="trading-bots-heading"
    >
      {error && runs === null ? (
        <EmptyState title="Bot status unavailable">{error}</EmptyState>
      ) : runs === null ? (
        <EmptyState title="Loading bot status" compact />
      ) : runs.length === 0 ? (
        <EmptyState title="No paper trading runs">
          Start one with: hft trade --paper --symbols NVDA=200 AAPL=100 --strategy ema-crossover
        </EmptyState>
      ) : (
        runs.map((run) => (
          <div
            key={run.name}
            className="table-scroll chart-table"
            tabIndex="0"
            role="region"
            aria-label={`Paper trading run ${run.name}`}
          >
            <table className="paper-sessions-table">
              <caption>Run {run.name}</caption>
              <thead>
                <tr>
                  <th scope="col">Stock</th>
                  <th scope="col">Status</th>
                  <th scope="col" className="numeric">
                    Budget
                  </th>
                  <th scope="col" className="numeric">
                    Position
                  </th>
                  <th scope="col" className="numeric">
                    Equity
                  </th>
                  <th scope="col" className="numeric">
                    Profit
                  </th>
                  <th scope="col">Order</th>
                </tr>
              </thead>
              <tbody>
                {run.stocks.map((stock) => (
                  <tr key={stock.symbol}>
                    <th scope="row">{stock.symbol}</th>
                    <td>{stock.status}</td>
                    <td className="numeric">{money(stock.budget)}</td>
                    <td className="numeric">{stock.position}</td>
                    <td className="numeric">{money(stock.equity)}</td>
                    <td className="numeric">{money(stock.profit)}</td>
                    <td>{stock.pending ? 'Open' : 'None'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))
      )}
      {error && runs !== null && <p role="status">Showing the last update: {error}</p>}
    </Panel>
  );
}
