import { statusLabel, statusTone } from './format';

export function Status({ value, label }) {
  return (
    <span className={`status status--${statusTone(value)}`}>
      <span className="status-dot" aria-hidden="true" />
      {label || statusLabel(value)}
    </span>
  );
}

export function Panel({ title, subtitle, actions, children, className = '', headingId }) {
  return (
    <section className={`panel ${className}`} aria-labelledby={headingId}>
      <div className="panel-header">
        <div>
          <h2 id={headingId}>{title}</h2>
          {subtitle && <p className="panel-subtitle">{subtitle}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function EmptyState({ title, children, compact = false }) {
  return (
    <div className={`empty-state ${compact ? 'empty-state--compact' : ''}`}>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
    </div>
  );
}
