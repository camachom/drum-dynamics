# Guardrails

## Debugging Tips

### 1. Test with a simple signal first
Before using real audio, create a fake signal with known hits:

```python
import numpy as np

sr = 44100
duration = 2.0
samples = np.zeros(int(sr * duration))

# Insert fake "hits" at known times
hit_times = [0.5, 1.0, 1.5]  # seconds
for t in hit_times:
    idx = int(t * sr)
    samples[idx:idx+100] = 0.8  # sharp spike

# Now test your functions - you should detect exactly 3 hits at those times
```

### 2. Visualize intermediate steps
When stuck, plot what you have (just temporarily):

```python
import matplotlib.pyplot as plt
plt.plot(envelope)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()
```

This will show you if your envelope looks reasonable and if your threshold makes sense.

### 3. Common pitfalls

| Problem | Likely cause |
|---------|--------------|
| Detecting 0 hits | Threshold too high, or envelope is all zeros |
| Detecting thousands of hits | Threshold too low, or no minimum distance between onsets |
| Amplitudes are all the same | Forgot to normalize audio, or using wrong dtype |
| Times are wrong | Forgot to account for window_size when converting indices |

## Milestones

- [ ] `load_audio` works: you can print the shape and duration
- [ ] `compute_envelope` works: envelope is shorter than samples, values are positive
- [ ] `detect_onsets` works on synthetic signal: finds the exact hits you inserted
- [ ] `detect_onsets` works on real audio: reasonable number of hits (not 0, not 10000)
- [ ] `measure_hits` gives times that match when you hear hits in the recording
- [ ] Amplitudes vary: ghost notes are quieter than accents

## When you're stuck

1. Print shapes: `print(samples.shape, envelope.shape)`
2. Print ranges: `print(samples.min(), samples.max())`
3. Test with synthetic signal where you know the answer
4. Plot it

## Don't overcomplicate it

A working V0 can be ~50 lines of actual code. If you're over 100, you're probably overengineering.
