"""
Drum Dynamics Analyzer - Skeleton
==================================

Your goal: Extract hit times and amplitudes from a drum recording.

Dependencies you'll need:
    pip install numpy scipy

Optional (for loading various audio formats):
    pip install soundfile

Suggested order of implementation:
    1. load_audio - get samples into numpy array
    2. compute_envelope - smooth the signal to see energy over time
    3. detect_onsets - find where hits occur
    4. measure_hits - get amplitude at each hit
    5. analyze (stretch goal) - answer your interesting questions
"""

from curses import KEY_OPTIONS
import numpy as np
from scipy.io import wavfile


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return (samples, sample_rate).

    Hints:
    - scipy.io.wavfile.read() returns (rate, data)
    - If stereo, you'll get shape (N, 2) - convert to mono by averaging channels
    - Normalize to float in range [-1, 1] (divide by max value for dtype)

    Returns:
        samples: 1D numpy array of floats
        sample_rate: int (e.g., 44100)
    """
    rate, data = wavfile.read(filepath)
    original_dtype = data.dtype

    if data.ndim > 1:
        data = data.mean(axis=1)

    floats = data.astype(np.float32)
    samples = floats / np.iinfo(original_dtype).max
    return samples, rate

def compute_envelope(samples: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """
    Compute the amplitude envelope of the signal.

    Hints:
    - RMS (root mean square) in sliding windows is a common approach
    - np.convolve can help, or just loop through windows
    - Consider: should windows overlap? By how much?

    Args:
        samples: audio samples
        window_size: number of samples per window (try 1024 for ~23ms at 44.1kHz)

    Returns:
        envelope: 1D array, one value per window (will be shorter than samples)
    """

    envelope = np.zeros(len(samples) // window_size)
    for i in range(0, len(samples) - window_size, window_size):
        window = samples[i:i + window_size]

        index = i // window_size
        envelope[index] = np.sqrt(np.mean(window**2))

    return envelope


def detect_onsets(envelope: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Find indices in the envelope where hits occur.

    TODO: Implement this.

    Hints:
    - Simple approach: find where envelope crosses a threshold
    - Better: find where envelope RISES above threshold (first-order difference > 0)
    - Problem: one hit might cross threshold multiple times (ringing)
      → Consider a "minimum distance" between detected onsets
    - If threshold is None, try computing one from the signal (e.g., mean * 1.5)

    Args:
        envelope: amplitude envelope from compute_envelope()
        threshold: minimum amplitude to count as a hit

    Returns:
        onset_indices: indices into the envelope array where hits occur
    """
    if threshold is None:
        threshold = np.mean(envelope) * 1.5

    diffs = np.diff(envelope)
    rising = diffs > 0
    above = envelope[1:] > threshold

    hits = np.where(rising & above)[0]

    if len(hits) == 0:
      return np.array([], dtype=int)

    hits = np.sort(hits)
    kept = [hits[0]]
    last = hits[0]
    min_distance = 3

    for h in hits[1:]:
        if h - last >= min_distance:
            kept.append(h)
            last = h

    return np.array(kept, dtype=int)

def measure_hits(samples: np.ndarray, onset_indices: np.ndarray,
                 window_size: int, sample_rate: int) -> list[dict]:
    """
    Measure the amplitude of each detected hit.

    TODO: Implement this.

    Hints:
    - onset_indices are positions in the ENVELOPE, not the raw samples
    - Convert: sample_position = onset_index * window_size (approximately)
    - For each onset, look at a small window and find peak amplitude
    - Return both time (in seconds) and amplitude

    Args:
        samples: original audio samples
        onset_indices: from detect_onsets()
        window_size: same window_size used in compute_envelope
        sample_rate: for converting to seconds

    Returns:
        List of {"time": float, "amplitude": float} dicts
    """
    raise NotImplementedError("Your turn")


# =============================================================================
# STRETCH GOALS - Implement these after the basics work
# =============================================================================

def normalize_amplitudes(hits: list[dict]) -> list[dict]:
    """
    Normalize hit amplitudes to range [0, 1].

    Why? Makes it easier to compare across recordings with different gain.

    Hint: Find max amplitude, divide all by it.
    """
    raise NotImplementedError("Stretch goal")


def classify_dynamics(hits: list[dict]) -> list[dict]:
    """
    Add a 'dynamic' label to each hit: 'ghost', 'normal', 'accent'.

    Hints:
    - Ghost notes: quietest 20%? Below some threshold?
    - Accents: loudest 20%? Above some threshold?
    - This is subjective - experiment!
    """
    raise NotImplementedError("Stretch goal")


def detect_tempo_changes(hits: list[dict]) -> list[dict]:
    """
    Compute instantaneous tempo at each hit.

    Hint: tempo = 60 / inter_onset_interval (gives BPM)
    """
    raise NotImplementedError("Stretch goal")


# =============================================================================
# MAIN - Use this to test as you go
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dynamics.py <audio_file.wav>")
        sys.exit(1)

    filepath = sys.argv[1]

    # Uncomment these as you implement each function:

    samples, sr = load_audio(filepath)
    print(f"Loaded {len(samples)} samples at {sr}Hz ({len(samples)/sr:.1f} seconds)")

    envelope = compute_envelope(samples, window_size=1024)
    print(f"Envelope has {len(envelope)} frames")

    onsets = detect_onsets(envelope)
    print(f"Detected {len(onsets)} hits")

    # hits = measure_hits(samples, onsets, window_size=1024, sample_rate=sr)
    # for h in hits[:10]:  # Print first 10
    #     print(f"  {h['time']:.3f}s: {h['amplitude']:.4f}")
