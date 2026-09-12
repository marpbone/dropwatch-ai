# An honest assessment of the original brief

You asked to be told what in the plan was unrealistic, unnecessary, or better
done another way. Here it is, roughly in order of how much time it will save you.

## Things in the brief that are wrong or will waste your time

### "C++ for performance-critical DSP if justified"

It is not justified for offline analysis, and adding it for its own sake will
cost you weeks and make the project *look worse*, not better. NumPy already
dispatches to compiled BLAS and FFTW-class kernels; a hand-written C++ STFT will
almost certainly be slower than `librosa.stft`, and a reviewer who knows this
will read the rewrite as not understanding where time actually goes.

There is a defensible C++ story here, but it is a different one. Two options:

1. **A real-time preview engine.** Offline analysis is a batch job; a JUCE or
   PortAudio component that plays the track and fires cue points live, with
   sample-accurate loop rolls, is genuinely a C++/real-time-audio problem —
   lock-free ring buffers, no allocation on the audio thread, callback deadlines.
   That demonstrates real-time engineering. Offline DSP does not.
2. **One measured hot spot.** Profile first, then rewrite what dominates. In this
   codebase that was HPSS at ~85% of analysis time — and the right fix turned out
   to be running it on the 128-band mel spectrogram instead of the 1025-bin
   linear one, a 20× speedup in *Python*. That is the more valuable lesson and
   it is now documented in `audio/features.py`.

Do not write C++ until you can point at a profile and say "this is 60% of
runtime and it is not vectorisable."

### "Machine learning" as a goal rather than a tool

The brief lists PyTorch under things to demonstrate. Resist this. There is no
public dataset of "drop / breakdown / buildup" annotations at a useful scale, so
you would be training on data you annotated yourself — a few hundred examples at
best, which is far too few for a neural section classifier and exactly the right
amount for a logistic regression.

Where ML genuinely earns its place in this project:

- **Source separation for vocals** (Demucs). Use a pretrained model; do not train
  one. This is a real accuracy win over any heuristic, and it is already wired in
  behind `VocalDetector`.
- **The confidence model refit from user feedback.** Small, honest, and it is the
  *only* part of the system with a natural labelled dataset — because the editor
  generates one every time a DJ accepts, rejects or drags a cue. That is a
  genuine human-in-the-loop ML story and it is more impressive than a
  fine-tuned transformer nobody can evaluate.
- **A learned beat/downbeat tracker**, if you want one — but as a *comparison
  baseline* against the DSP one, evaluated properly. "I built both and measured
  them" is a stronger claim than "I used a model".

Saying "this project uses classical DSP where DSP is better and ML where ML is
better, and here is the measurement that decided each case" is a far better
interview answer than a longer technology list.

### "Detect song sections such as intro, verse, buildup, breakdown, drop, chorus, outro"

Fine as a goal, but understand what is actually hard. Finding *boundaries* is
tractable and largely solved (self-similarity + novelty). Assigning *names* is
not, because the names are conventions, not acoustic facts — a "drop" in techno
and a "chorus" in pop can be spectrally identical, and plenty of tracks have
neither. Expect boundary detection to work well and labelling to be the part that
embarrasses you on real music.

Two consequences: `UNKNOWN` must be a legitimate output, and you should measure
boundaries and labels *separately* — a system with F1 0.9 on boundaries and 60%
on labels is useful; the single blended number would hide that.

### Essentia

Listed among the technologies. It is excellent and its `RhythmExtractor2013` is
genuinely good, but installation is historically painful (compiled dependencies,
patchy wheels) and it will eat a day you do not have. librosa + scipy covers
everything in the MVP. Add Essentia later, as an alternative backend behind the
same interface, if and only if you have measured that you need it.

### "Detect BPM and beat grid" as one item

These are not one item. Tempo estimation is easy and mostly solved. **Downbeat
phase is the hard part and the brief does not mention it at all** — yet getting
it wrong by one beat ruins every phrase-aligned cue downstream, which is most of
the value of the tool. Budget real time for it. It was the single most
troublesome part of building this, and the first three cue designs I tried voted
confidently for the *wrong* answer.

## Things the brief underrates

### The export layer is the most interesting engineering in the project

The brief treats "don't couple to Rekordbox" as hygiene. It is actually the best
system-design story you have: capability declarations, a lowering pass, a
perceptual colour-quantisation step, and a report of everything the target format
cannot represent. It is small, it is finishable, and it is the part that shows you
can design an interface rather than just implement one. Lead with it.

### Explanations are worth more than confidence scores

The brief asks for a confidence score *and* reasons. The reasons are the valuable
half. A number nobody can interrogate is not trustworthy; "84%, because it is
preceded by a 16-bar build-up, lands on a phrase boundary, and the bass enters
here" is. Building the model so the explanation *is* the computation — rather
than prose generated alongside it — is the design decision that makes this true,
and it is why the scorer is a linear model.

### Hot cue *ordering* is a real requirement nobody writes down

Rekordbox gives you eight hot cue slots. Which cues get them is a question of
importance. Which slot each gets is a question of ergonomics, and the answer is
always chronological, because pads A–H run left to right under the DJ's hand.
Conflating the two produces a technically correct export that is unusable in a
club. My first implementation did exactly this.

## Things you should cut from v1

- **Serato / Traktor / Engine DJ export.** Design the interface so they are
  possible; implement only Rekordbox. Two targets prove the abstraction (JSON is
  the second); four is busywork.
- **Key detection and harmonic mixing.** Real, useful, and a completely separate
  project. `key` is a nullable field in the model; leave it null.
- **User accounts, multi-user, cloud storage.** Single-user local tool. Nobody
  reviewing this cares about your auth.
- **Playlist / crate management.** You are building a track preparation tool, not
  a library manager.
- **The `Chorus`, `Bridge` and `Verse` labels**, unless you play vocal-led music.
  They need reliable vocal detection to be meaningful and they add failure modes.

## The honest risk

The single biggest risk is that you build all of this and it produces *plausible
but slightly wrong* cues on real tracks — 6 bars off, or a "drop" on the second
breakdown — and you have no way to tell how wrong because you never built the
evaluation set. Everything else is recoverable; that is not.

Annotate 40 tracks before you write the recommender. Two hours of work, and it is
the difference between a project you can make claims about and one you cannot.
