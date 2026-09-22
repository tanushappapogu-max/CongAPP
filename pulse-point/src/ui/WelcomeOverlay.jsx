import { useEffect, useRef } from 'react';

export default function WelcomeOverlay({ onDismiss }) {
  const dialogRef = useRef(null);
  const startButtonRef = useRef(null);
  const dismissRef = useRef(onDismiss);
  dismissRef.current = onDismiss;

  useEffect(() => {
    const previousFocus = document.activeElement;
    startButtonRef.current?.focus();

    const onKey = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        dismissRef.current();
        return;
      }

      if (event.key !== 'Tab' || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll(
        'button, input, select, textarea, [href], [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable.length) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      const restore = previousFocus && previousFocus !== document.body
        ? previousFocus
        : document.getElementById('target-input');
      restore?.focus?.();
    };
  }, []);

  return (
    <div className="welcome-overlay">
      <section
        ref={dialogRef}
        className="welcome-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="welcome-title"
        aria-describedby="welcome-description"
      >
        <p className="welcome-kicker">PULSE POINT / FIELD GUIDE</p>
        <h2 id="welcome-title">Find what’s near you.</h2>
        <p id="welcome-description" className="welcome-tagline">
          Find nearby objects through your camera, vibration, and voice.
        </p>

        <ol className="welcome-steps">
          <li>
            <span className="welcome-step-number">01</span>
            <span><strong>Speak or type</strong> what you want to find.</span>
          </li>
          <li>
            <span className="welcome-step-number">02</span>
            <span><strong>Point your camera</strong> at the scene.</span>
          </li>
          <li>
            <span className="welcome-step-number">03</span>
            <span><strong>Follow haptics and voice</strong> toward the target.</span>
          </li>
        </ol>

        <p className="welcome-permission" role="note">
          This app needs your camera and microphone to scan and listen for targets.
          You can change guidance preferences in Settings.
        </p>

        <button
          ref={startButtonRef}
          type="button"
          className="welcome-start"
          onClick={onDismiss}
        >
          Get started
        </button>
      </section>
    </div>
  );
}
