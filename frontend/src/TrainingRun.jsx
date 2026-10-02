import Icon from './Icon';
import { number, trialProgress } from './format';
import { Panel, Status } from './primitives';

export default function TrainingRun({ training }) {
  const progress = trialProgress(training);
  const stages = [
    {
      name: 'Data validated',
      description: training.data_validated
        ? 'Dataset loaded and checks completed.'
        : training.status === 'insufficient-data'
          ? 'Development coverage checks have not passed.'
          : 'Awaiting completed dataset checks.',
      complete: training.data_validated,
    },
    {
      name: 'Model training',
      description:
        training.status === 'running'
          ? 'Search is running on development sessions.'
          : training.status === 'interrupted'
            ? 'Interrupted during training.'
            : progress.total !== null && progress.completed === progress.total
              ? 'Research search completed.'
              : 'Search has not completed.',
      complete: progress.total !== null && progress.completed === progress.total,
      current: ['running', 'interrupted', 'incomplete-search'].includes(training.status),
    },
    {
      name: 'Held-out evaluation',
      description: training.held_out_evaluated
        ? 'Final-test evaluation recorded. Review the report.'
        : 'Reserved for after successful training.',
      complete: training.held_out_evaluated,
    },
    {
      name: 'Paper eligibility',
      description: training.paper_eligible
        ? 'Research eligibility recorded; paper evidence still required.'
        : 'Awaiting validation and completed evaluation.',
      complete: training.paper_eligible,
    },
  ];
  return (
    <Panel
      title="Training run"
      actions={<Status value={training.status} />}
      className="training-panel"
      headingId="training-heading"
    >
      <p className="training-message">
        {training.message || 'No training status is available yet.'}
      </p>
      <ol className="stages">
        {stages.map((stage) => (
          <li
            key={stage.name}
            className={stage.complete ? 'stage--complete' : stage.current ? 'stage--current' : ''}
          >
            <span className="stage-marker" aria-hidden="true">
              {stage.complete ? (
                <Icon name="check" />
              ) : stage.current ? (
                <Icon name="pause" />
              ) : null}
            </span>
            <div>
              <h3>{stage.name}</h3>
              <p>{stage.description}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="trial-progress">
        {progress.percent !== null ? (
          <progress
            max={progress.total}
            value={Math.min(progress.completed, progress.total)}
            aria-label="Completed research trials"
          />
        ) : (
          <div className="progress-unavailable" aria-hidden="true" />
        )}
        <span>
          {number(progress.completed)} / {number(progress.total)} completed trials
        </span>
      </div>
      {Number.isFinite(training.current_trial) && (
        <p className="run-current">
          Current trial: {number(training.current_trial)}
          {Number.isFinite(training.steps_completed)
            ? ` · ${number(training.steps_completed)} steps recorded`
            : ''}
        </p>
      )}
    </Panel>
  );
}
