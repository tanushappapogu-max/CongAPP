import { Speaker } from './speech.js';

class FakeUtterance {
  constructor(text) { this.text = text; }
}

function installSynth() {
  const synth = {
    spoken: [],
    speak: vi.fn(u => synth.spoken.push(u)),
    cancel: vi.fn(),
    resume: vi.fn(),
    getVoices: () => [],
    addEventListener: () => {},
  };
  window.speechSynthesis = synth;
  globalThis.SpeechSynthesisUtterance = FakeUtterance;
  return synth;
}

describe('Speaker', () => {
  let synth;
  beforeEach(() => {
    vi.useFakeTimers();
    synth = installSynth();
  });
  afterEach(() => {
    vi.useRealTimers();
    delete window.speechSynthesis;
    delete globalThis.SpeechSynthesisUtterance;
  });

  test('stays silent until enabled', () => {
    const s = new Speaker();
    s.say('left');
    expect(synth.speak).not.toHaveBeenCalled();
    s.setEnabled(true);
    s.say('left');
    expect(synth.spoken.map(u => u.text)).toEqual(['left']);
  });

  test('prime speaks the given phrase and sets the iOS audio session to playback', () => {
    navigator.audioSession = { type: 'auto' };
    const s = new Speaker();
    s.setEnabled(true);
    s.prime('Starting.');
    expect(synth.spoken.map(u => u.text)).toEqual(['Starting.']);
    expect(navigator.audioSession.type).toBe('playback');
    delete navigator.audioSession;
  });

  test('prime without a phrase does not queue an empty utterance', () => {
    const s = new Speaker();
    s.setEnabled(true);
    s.prime();
    expect(synth.speak).not.toHaveBeenCalled();
    expect(s.primed).toBe(true);
  });

  test('queues while speaking and plays the pending phrase after onend', () => {
    const s = new Speaker();
    s.setEnabled(true);
    s.say('left');
    s.say('right');
    expect(synth.spoken).toHaveLength(1);
    synth.spoken[0].onend();
    vi.advanceTimersByTime(600);
    expect(synth.spoken.map(u => u.text)).toEqual(['left', 'right']);
  });

  test('recovers when the browser drops an utterance and never fires onend', () => {
    const s = new Speaker();
    s.setEnabled(true);
    s.say('left');
    s.say('right'); // queued behind a ghost utterance
    vi.advanceTimersByTime(10_000);
    expect(synth.spoken.map(u => u.text)).toEqual(['left', 'right']);
  });

  test('forced phrase waits for cancel to settle and ignores the old onend', () => {
    const s = new Speaker();
    s.setEnabled(true);
    s.say('left');
    const first = synth.spoken[0];
    s.say('reach', { urgent: true, force: true });
    expect(synth.cancel).toHaveBeenCalled();
    expect(synth.spoken).toHaveLength(1); // not spoken in the same tick as cancel()
    first.onend(); // late callback from the cancelled utterance
    vi.advanceTimersByTime(100);
    expect(synth.spoken.map(u => u.text)).toEqual(['left', 'reach']);
    expect(s.isSpeaking).toBe(true);
  });

  test('keeps waiting while the engine reports it is still speaking', () => {
    const s = new Speaker();
    s.setEnabled(true);
    synth.speaking = true;
    s.say('a long phrase');
    s.say('next');
    vi.advanceTimersByTime(5000);
    expect(synth.spoken).toHaveLength(1);
    synth.speaking = false;
    vi.advanceTimersByTime(1600);
    expect(synth.spoken.map(u => u.text)).toEqual(['a long phrase', 'next']);
  });

  test('skips novelty and network voices', () => {
    synth.getVoices = () => [
      { name: 'Albert', lang: 'en-US', localService: true },
      { name: 'Google US English', lang: 'en-US', localService: false },
      { name: 'Samantha', lang: 'en-US', localService: true },
    ];
    expect(new Speaker().voice.name).toBe('Samantha');
  });

  test('recovers if speak() throws', () => {
    synth.speak.mockImplementationOnce(() => { throw new Error('blocked'); });
    const s = new Speaker();
    s.setEnabled(true);
    s.say('left');
    expect(s.isSpeaking).toBe(false);
  });
});
