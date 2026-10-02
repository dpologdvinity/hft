import { Component } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

class DashboardBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed)
      return (
        <main className="fatal-state">
          <h1>Dashboard could not be displayed</h1>
          <p>The local response could not be shown. Reload to request a new snapshot.</p>
          <button type="button" className="primary-button" onClick={() => window.location.reload()}>
            Reload dashboard
          </button>
        </main>
      );
    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(
  <DashboardBoundary>
    <App />
  </DashboardBoundary>,
);
