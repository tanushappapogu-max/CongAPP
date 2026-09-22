import { useEffect, useRef } from 'react';

/**
 * StartupProgress
 *
 * A full-screen boot HUD that shows the exact startup steps the user
 * must wait through (camera, download, compile, warmup) with a live
 * progress bar and honest status text.
 *
 * Props:
 *   steps      — array of step descriptors (provided by App.jsx)
 *   overallPct — 0-100, drives the main progress bar
 *   onCancel   — called when the user taps "Cancel"
 *   isFirstBoot — boolean, shown in the bottom hint
 */
export default function StartupProgress({ steps = [], overallPct = 0, onCancel, isFirstBoot = false }) {
  const cancelRef = useRef(null);

  // Keep focus on the cancel button so screen readers track it.
  useEffect(() => {
    cancelRef.current?.focus();
  }, []);

  const clamp = v => Math.max(0, Math.min(100, Math.round(v)));
  const pct = clamp(overallPct);

  return (
    <div className="startup-overlay" role="status" aria-live="polite" aria-label="Starting Pulse Point">
      <div className="startup-card">

        {/* ── Header ── */}
        <div className="startup-header">
          <span className="startup-badge">PULSE POINT</span>
          <p className="startup-title">Starting engine…</p>
        </div>

        {/* ── Overall progress bar ── */}
        <div className="startup-bar-track" aria-hidden="true">
          <div
            className="startup-bar-fill"
            style={{ width: `${pct}%` }}
          />
          <span className="startup-bar-pct">{pct}%</span>
        </div>

        {/* ── Step list ── */}
        <ol className="startup-steps" aria-label="Boot sequence">
          {steps.map(step => (
            <li key={step.id} className={`startup-step startup-step--${step.state}`}>
              <span className="startup-step-icon" aria-hidden="true">
                {step.state === 'done'    && '✓'}
                {step.state === 'active'  && <span className="startup-spinner" />}
                {step.state === 'pending' && '○'}
                {step.state === 'error'   && '✗'}
              </span>
              <span className="startup-step-body">
                <span className="startup-step-label">{step.label}</span>
                {step.detail && (
                  <span className="startup-step-detail">{step.detail}</span>
                )}
                {/* Inline sub-bar for the download step */}
                {step.state === 'active' && step.subPercent != null && (
                  <div className="startup-sub-bar-track" aria-hidden="true">
                    <div
                      className="startup-sub-bar-fill"
                      style={{ width: `${clamp(step.subPercent)}%` }}
                    />
                  </div>
                )}
              </span>
            </li>
          ))}
        </ol>

        {/* ── Hint text ── */}
        <p className="startup-hint">
          {isFirstBoot
            ? 'Downloading the 12.8 MB neural network. Cached locally — future starts take under a second.'
            : 'Loading cached model and spinning up the inference engine.'}
        </p>

        {/* ── Cancel ── */}
        {onCancel && (
          <button
            ref={cancelRef}
            type="button"
            className="startup-cancel"
            onClick={onCancel}
            aria-label="Cancel startup"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
