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

import numpy as np
from scipy.io import wavfile


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Audio file can have more than one channel (stereo). This is an example of `data` from a stereo file:

    # a.shape
    # (num of samples, num of channels)
    (85441, 2)

    # print(a) 
    [[      0       0]
    [ -65536  -65536]
    [-917504 -917504]
    ...]

    Take the mean for each row and collapse this into a single array:
    `data.mean(axis=1)`

    The rest is convention. Seems like mapping amplitudes to [-1,1] is standard.
    """
    rate, data = wavfile.read(filepath)
    original_dtype = data.dtype

    print(data.shape)
    print(data)


    if data.ndim > 1:
        data = data.mean(axis=1)

    floats = data.astype(np.float32)
    samples = floats / np.iinfo(original_dtype).max
    return samples, rate

def compute_envelope(samples: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """
    If you plot the sample, its really jagged and chaotic. Envelops are used
    to more easily work with data. They basically smooth out the curve.

    Given a window size, use the RMS (root mean square) algorithm and add it to
    this new `envelop` array.

    Intuitively, you want the mean without all those sudden drops.
    """

    envelope = np.zeros(len(samples) // window_size)
    for i in range(0, len(samples) - window_size, window_size):
        window = samples[i:i + window_size]

        index = i // window_size
        envelope[index] = np.sqrt(np.mean(window**2))

    return envelope


def detect_onsets(envelope: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Threshold will determine if a sample is loud enough to be a `hit` candidate.

    `np.diff` computes the difference between elements in the `envelop` array. A `hit`
    needs to be rising (so diff is positive) and above the threshold. An diff that's
    negative is just sustain from a previous `hit`.

    The `hits` array is a set of indexes that satisfy the criteria. In order to avoid
    counting the same hit multiple times, there's a minimum distance of 3 indexes. 
    """
    if threshold is None:
        threshold = np.mean(envelope) * 1.2

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
    Sample rate is samples per second (e.g., 44100 Hz = 44100 samples/sec).
    We need to find the timestamp of when the hit occurred (aka seconds).
    
    SR = samples / sec
    SR * sec = samples
    sec = samples / SR

    This might sound silly but its a good reminder that a negative value could be the peak. 
    Loudness is measured in displacement from 0 in either direction (which is why it uses np.abs). 
    """
    return [
      {
          "time": (onset_idx * window_size) / sample_rate,
          "amplitude": np.max(np.abs(samples[onset_idx * window_size : (onset_idx + 1) * window_size]))
      }
      for onset_idx in onset_indices
    ]


# =============================================================================
# STRETCH GOALS - Implement these after the basics work
# =============================================================================

def normalize_amplitudes(hits: list[dict]) -> list[dict]:
    """
    Normalize hit amplitudes to range [0, 1].

    Why? Makes it easier to compare across recordings with different gain.

    Hint: Find max amplitude, divide all by it.
    """
    max_amplitude = max(h["amplitude"] for h in hits)
    return [
        {"time": h["time"], "amplitude": h["amplitude"] / max_amplitude}
        for h in hits
    ]


def classify_dynamics(hits: list[dict]) -> list[dict]:
    """
    Add a 'dynamic' label to each hit: 'ghost', 'normal', 'accent'.

    Only classifies if there's meaningful variation in amplitudes.
    """
    amplitudes = [h["amplitude"] for h in hits]
    mean_amp = np.mean(amplitudes)
    std_amp = np.std(amplitudes)

    # If all hits are similar (low variance), call them all normal
    if std_amp < mean_amp * 0.15:
        return [{**h, "dynamic": "normal"} for h in hits]

    ghost_threshold = np.percentile(amplitudes, 20)
    accent_threshold = np.percentile(amplitudes, 80)

    return [
        {**h, "dynamic": calculate_dynamic(h["amplitude"], ghost_threshold, accent_threshold)}
        for h in hits
    ]

def calculate_dynamic(amplitude: float, ghost_percentile: float, accent_percentile: float) -> str:
    if amplitude <= ghost_percentile:
        return 'ghost'
    elif amplitude >= accent_percentile:
        return 'accent'
    else:
        return 'normal'


def detect_tempo_changes(hits: list[dict]) -> list[dict]:
    """
    Compute instantaneous tempo at each hit.

    Hint: tempo = 60 / inter_onset_interval (gives BPM)
    """
    times = [h["time"] for h in hits]
    tempos = 60 / np.diff(times)

    result = [
        {**h, "tempo": tempos[i]}
        for i, h in enumerate(hits[:-1])
    ]
    result.append({**hits[-1], "tempo": None})
    return result


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

    hits = measure_hits(samples, onsets, window_size=1024, sample_rate=sr)
    for h in hits[:10]:  # Print first 10
        print(f"  {h['time']:.3f}s: {h['amplitude']:.4f}")

    # Stretch goals:

    normalized = normalize_amplitudes(hits)
    print(f"\nNormalized amplitudes:")
    for h in normalized[:10]:
        print(f"  {h['time']:.3f}s: {h['amplitude']:.4f}")

    classified = classify_dynamics(normalized)
    print(f"\nClassified dynamics:")
    for h in classified[:10]:
        print(f"  {h['time']:.3f}s: {h['amplitude']:.4f} ({h['dynamic']})")

    with_tempo = detect_tempo_changes(classified)
    print(f"\nTempo changes:")
    for h in with_tempo[:10]:
        print(f"  {h['time']:.3f}s: {h.get('tempo', 'N/A')} BPM")