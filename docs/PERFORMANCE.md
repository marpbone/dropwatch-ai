# Performance

Analysis of a 5-minute track, single core:

| Stage | Time | Notes |
|---|---|---|
| decode + resample | 1.5 s | ffmpeg/libsndfile → mono 22.05 kHz |
| feature extraction | 5.1 s | one STFT, everything derived from it |
| beat grid | 3.8 s | coarse tempo scan + fine (bpm, phase) refinement |
| phrase grid | 0.07 s | |
| vocals (heuristic) | 0.02 s | separation mode is ~0.5–2× realtime instead |
| boundaries + merge | 0.05 s | |
| labelling | 0.001 s | rules + Viterbi over ~10 segments |
| **total** | **~10 s** | |
| re-recommend after a config change | **~0.04 s** | no DSP re-run |

## The optimisation that mattered

A naive first implementation spent ~85% of analysis time in
`librosa.decompose.hpss` at full spectrogram resolution: 3.9 s of a 4.6 s feature
pass on a 60-second excerpt.

Harmonic/percussive separation is two median filters over the spectrogram, so cost
scales with its size. We only use the result for (a) a scalar percussive-ratio per
frame and (b) a harmonic mask in the 180 Hz–4 kHz vocal region. Neither needs 1025
linear frequency bins; 128 mel bands are finer than the decision being made from
them.

Moving HPSS to the mel domain: **20× faster**, no measurable change in downstream
output. Feature extraction went from ~2× realtime to ~78× realtime.

The general lesson, and the reason this file exists: the fix was not writing the
median filter in C++. It was noticing the operation was being run at a resolution
the answer did not require. Profile before optimising, and question the problem
size before questioning the language.

## Where the remaining time goes

`beats` (3.8 s) is dominated by the fine (bpm, phase) lattice search. It is already
reduced from a naive O(n_phases × n_beats) brute force to O(T) per tempo candidate
by **phase folding** — histogramming onset energy modulo the beat period, so the
peak bin *is* the best phase. Further gains would come from narrowing the candidate
set, not from a faster inner loop.

`features` (5.1 s) is dominated by the STFT itself and is essentially at the floor
for this frame rate.

## A note on the first measurement

The very first timing run reported 21.6 s for 10 seconds of audio — apparently 2×
slower than realtime. Almost all of it was numba JIT warm-up inside librosa on
first call. Always warm up before timing, and be suspicious of a single cold
measurement.

## If you did want to justify C++

Not here. Both hot spots are already vectorised NumPy dispatching to compiled
kernels. See `docs/CRITIQUE.md` for where a C++ component genuinely belongs in
this project: a real-time playback and preview engine, which is a different
problem with hard deadlines, rather than an offline batch job.
