// Spoken guidance via the browser's SpeechSynthesis API.
//
// For users who can't reliably feel haptics (iPhone Safari has weak Vibration
// API support, certain Android phones suppress short vibrations) speech is the
// crucial alternative output channel. Even with haptics, speech adds the
// "what" + "how far" context that vibration can't carry.
//
// Behavior:
//   - Queues utterances to prevent overlapping by waiting for current speech to finish.
//   - Only interrupts for truly urgent signals (reach, closer).
//   - Tracks speaking state and plays pending speech when current finishes.
//   - Throttles repetition of the same phrase to MIN_GAP_MS unless `urgent`.
//   - Prefers an on-device English voice, otherwise the system default.
//
// Mobile quirks handled here:
//   - iOS Safari and Chrome ignore speak() until it has been called inside a
//     user gesture → prime() must run synchronously from a tap handler and
//     speak a real phrase (iOS ignores utterance.volume, and an empty utterance
//     can wedge the iOS queue).
//   - iOS mutes speech when the ringer switch is on silent unless the page's
//     audio session is set to 'playback' (Safari 16.4+).
//   - Chrome (notably Android) drops a speak() issued in the same tick as
//     cancel(), so forced phrases wait CANCEL_SETTLE_MS after cancelling.
//   - Browsers sometimes drop an utterance or never fire onend; a watchdog
//     polls the engine so the queue can't get stuck behind a ghost.

const MIN_GAP_MS = 2000;
const URGENT_GAP_MS = 600;

// Silent pause inserted between consecutive spoken instructions so they don't
// run into each other. Urgent signals get a shorter pause to stay responsive.
const INTER_SPEECH_PAUSE_MS = 500;
const URGENT_PAUSE_MS = 150;

const CANCEL_SETTLE_MS = 80;
const WATCHDOG_POLL_MS = 1000;
const MAX_UTTERANCE_MS = 15000;

// Apple's novelty voices are tagged en-US but are not suitable for guidance.
const NOVELTY_VOICES = /Albert|Bad News|Bahh|Bells|Boing|Bubbles|Cellos|Good News|Jester|Organ|Superstar|Trinoids|Whisper|Wobble|Zarvox/i;

export class Speaker {
  constructor() {
    this.lastSaid = '';
    this.lastTime = 0;
    this.enabled = false;
    this.rate = 1.15;
    this.voice = null;
    this.voiceCount = 0;
    this.isSpeaking = false;
    this.pendingText = null;
    this.pendingUrgent = false;
    this.primed = false;
    this.lastError = '';
    // Strong reference to the in-flight utterance: Chrome can GC an unreferenced
    // utterance mid-speech and never fire onend. Also used to ignore callbacks
    // from utterances that were cancelled and replaced.
    this.current = null;
    this.watchdog = null;
    this.settleTimer = null;
    this._tryPickVoice();
    if (this.isAvailable()) {
      // voice list loads async on first call in most browsers
      const synth = window.speechSynthesis;
      if (synth.addEventListener) synth.addEventListener('voiceschanged', () => this._tryPickVoice());
      else synth.onvoiceschanged = () => this._tryPickVoice();
    }
  }

  isAvailable() {
    return typeof window !== 'undefined' && 'speechSynthesis' in window;
  }

  setEnabled(on) {
    this.enabled = on && this.isAvailable();
    if (!on) this.cancel();
  }

  setRate(r) {
    this.rate = Math.max(0.5, Math.min(2.5, r || 1));
  }

  /**
   * Unlock speech output. Call synchronously from a tap/click handler, before
   * any await. With `text`, that phrase is spoken immediately as the unlock.
   * @param {string} [text]
   */
  prime(text) {
    if (!this.isAvailable()) return;
    try {
      // Play through the iOS silent switch. Re-applied on every prime because
      // speech recognition switches the session to recording.
      if (typeof navigator !== 'undefined' && navigator.audioSession) {
        navigator.audioSession.type = 'playback';
      }
    } catch { /* unsupported */ }
    try {
      window.speechSynthesis.resume?.();
    } catch { /* unsupported */ }
    if (!this.voice) this._tryPickVoice();
    if (text && this.enabled) {
      this.lastSaid = '';
      this.say(text, { urgent: true, force: this.isSpeaking });
    }
    this.primed = true;
  }

  cancel() {
    if (this.isAvailable()) window.speechSynthesis.cancel();
    this._clearTimers();
    this.current = null;
    this.isSpeaking = false;
    this.pendingText = null;
  }

  /** Snapshot for the ?debug=1 overlay. */
  status() {
    const synth = this.isAvailable() ? window.speechSynthesis : null;
    return {
      enabled: this.enabled,
      primed: this.primed,
      voices: this.voiceCount,
      voice: this.voice?.name || 'default',
      speaking: Boolean(synth?.speaking),
      pending: Boolean(synth?.pending),
      lastError: this.lastError,
    };
  }

  /**
   * Speak a short phrase. Queues if currently speaking.
   * @param {string} text
   * @param {{urgent?: boolean, force?: boolean}} opts
   */
  say(text, { urgent = false, force = false } = {}) {
    if (!this.enabled || !this.isAvailable() || !text) return;
    const now = Date.now();
    const gap = urgent ? URGENT_GAP_MS : MIN_GAP_MS;

    // Don't repeat the same phrase too soon. This applies even to forced/urgent
    // signals: a continuously-firing signal like "closer" calls say() every
    // frame, and without this guard each call would cancel and restart the
    // in-progress utterance, producing "close-close-close-closer" stutter.
    if (text === this.lastSaid && now - this.lastTime < gap) return;

    // If the same phrase is still being spoken, never interrupt it with itself.
    if (this.isSpeaking && text === this.lastSaid) return;

    // If currently speaking, queue instead of interrupting (except for forced urgent)
    if (this.isSpeaking && !(force && urgent)) {
      this.pendingText = text;
      this.pendingUrgent = urgent;
      return;
    }

    let interrupted = false;
    if (force && urgent && this.isSpeaking) {
      window.speechSynthesis.cancel();
      this._clearTimers();
      this.pendingText = null;
      interrupted = true;
    }

    const u = new SpeechSynthesisUtterance(text);
    u.rate = this.rate;
    u.pitch = 1;
    u.volume = 1;
    u.lang = this.voice?.lang || 'en-US';
    if (this.voice) u.voice = this.voice;

    const finish = () => {
      // Ignore late callbacks from an utterance that was cancelled/replaced.
      if (this.current !== u) return;
      this._clearTimers();
      this.current = null;
      this.isSpeaking = false;
      this._playPending();
    };
    u.onstart = () => { if (this.current === u) this.isSpeaking = true; };
    u.onend = finish;
    u.onerror = (e) => {
      const err = e?.error || 'error';
      // 'interrupted'/'canceled' are our own cancel() calls, not failures.
      if (err !== 'interrupted' && err !== 'canceled') this.lastError = err;
      // Speech got locked again (e.g. page reloaded without a tap): re-prime on next tap.
      if (err === 'not-allowed') this.primed = false;
      finish();
    };

    // Mark speaking synchronously so a second say() before onstart fires
    // queues instead of starting an overlapping utterance.
    this.current = u;
    this.isSpeaking = true;
    this.lastSaid = text;
    this.lastTime = now;

    const start = () => {
      this.settleTimer = null;
      if (this.current !== u) return;
      try {
        window.speechSynthesis.resume?.(); // Chrome can leave the synth paused
        window.speechSynthesis.speak(u);
      } catch (e) {
        this.lastError = String(e?.message || e).slice(0, 60);
        finish();
        return;
      }
      this._armWatchdog(u, finish, Date.now());
    };

    if (interrupted) this.settleTimer = setTimeout(start, CANCEL_SETTLE_MS);
    else start();
  }

  _armWatchdog(u, finish, startedAt) {
    this.watchdog = setTimeout(() => {
      this.watchdog = null;
      if (this.current !== u) return;
      const synth = window.speechSynthesis;
      const busy = synth.speaking || synth.pending;
      if (!busy) { finish(); return; } // dropped, or onend lost
      if (Date.now() - startedAt > MAX_UTTERANCE_MS) {
        // Engine is wedged (seen on iOS); reset it so later phrases can play.
        synth.cancel();
        finish();
        return;
      }
      this._armWatchdog(u, finish, startedAt);
    }, WATCHDOG_POLL_MS);
  }

  _clearTimers() {
    if (this.watchdog) clearTimeout(this.watchdog);
    if (this.settleTimer) clearTimeout(this.settleTimer);
    this.watchdog = null;
    this.settleTimer = null;
  }

  _playPending() {
    if (this.pendingText) {
      const text = this.pendingText;
      const urgent = this.pendingUrgent;
      this.pendingText = null;
      this.pendingUrgent = false;
      // Wait out a short pause so the next instruction has a clear gap after
      // the previous one finishes instead of running straight into it.
      const pause = urgent ? URGENT_PAUSE_MS : INTER_SPEECH_PAUSE_MS;
      setTimeout(() => this.say(text, { urgent }), pause);
    }
  }

  _tryPickVoice() {
    if (!this.isAvailable()) return;
    const voices = (window.speechSynthesis.getVoices() || [])
      .filter(v => /^en/i.test(v.lang) && !NOVELTY_VOICES.test(v.name));
    this.voiceCount = voices.length;
    // Only pin on-device voices: network voices (some Android/Chrome builds)
    // fail silently when offline. With no good match, leave u.voice unset so
    // the platform default speaks.
    const local = voices.filter(v => v.localService !== false);
    this.voice =
      local.find(v => v.default)
      || local.find(v => /en[-_]US/i.test(v.lang) && /Samantha|Google|Microsoft|Alex/i.test(v.name))
      || local.find(v => /en[-_]US/i.test(v.lang))
      || null;
  }
}
