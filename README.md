# synthetic-pad

Synthetic data generation for training and evaluating face liveness detection models.

Undergraduate thesis (TCC) — Bachelor of Computer Science, Universidade Federal de Santa Catarina (UFSC).

---

## Overview

Face recognition has become one of the primary biometric authentication mechanisms, which makes **presentation attack detection (PAD)** — distinguishing a live person from a printed photo, a replayed video, or an artificially generated face — a critical security requirement.

Progress in this area is limited by the datasets available: real PAD datasets are small, lack diversity, and are constrained by privacy and legal restrictions on biometric data. Synthetic data offers a way around both problems, since generated samples do not correspond to real individuals and can be produced at scale.

This project investigates **whether synthetic presentation attacks can be generated well enough to be useful for evaluating and training liveness detection models**, and compares different generation strategies against each other.

## Research question

Can synthetic data be used to develop and evaluate face presentation attack detection models, and how do different generation strategies compare in terms of attack effectiveness and realism?

## Approach

The core experimental loop is deliberately simple:

```
bona fide image  ->  attack generator  ->  synthetic attack  ->  liveness detector  ->  score
```

The liveness detector acts as the **judge**: an attack is considered successful when the detector fails to flag it. This makes every generation strategy directly comparable under the same protocol.

### Generation strategies

Each generator implements the same interface, so strategies can be swapped and compared A/B:

| Strategy | Status | Description |
|---|---|---|
| Display recapture (moiré) | Baseline | Simulates the artifacts of recapturing a face from a screen — moiré interference, specular reflection, reduced dynamic range, geometric distortion |
| Diffusion-based | Planned | Generative synthesis of attack samples using diffusion models |
| VLM-conditioned | Planned | Attack generation conditioned through vision-language models |

#### The display recapture pipeline

A replay attack is a bona fide face shown on a screen and photographed again, so the
baseline generator models that optical path rather than stacking arbitrary
augmentations. The stages run in the order light actually travels, because the order
changes the result:

| # | Stage | What it models | Parameters |
|---|---|---|---|
| 1 | Geometry | The camera is never square-on to the screen, so the image arrives keystoned and slightly rolled | `tilt_x`, `tilt_y` (fractions of the perpendicular dimension), `rotation_deg` |
| 2 | Tone | The panel's tone curve and gamut replace the scene's: real replays come out more saturated and, on MSU-MFSD, contrastier than the bona fide captures | `contrast`, `saturation`, `black_level`, `white_level` |
| 3 | Glare | A light source in the room mirrored on the screen glass | `glare_intensity`, `glare_extent` (fraction of the image width) |
| 4 | Lens | Defocus and the finite pixel fill factor; off by default, since real replays measure sharper than bona fide | `blur_sigma_px` |
| 5 | Moiré | The screen's pixel grid beating against the sensor's | `moire_amplitude`, `moire_min_cycles_per_width`, `moire_max_cycles_per_width`, `moire_max_angle_deg`, `moire_chroma_shift_rad`, `moire_cross_weight` |
| 6 | Noise | Sensor read noise on a dim, indirect subject, in brightness only: video codecs keep colour at a quarter of the resolution, which smooths colour noise away | `noise_sigma_levels` |

The ordering is load-bearing. Glare lies on the glass, physically in front of the lens,
so stage 4 blurs it — a crisp highlight would read as a sticker rather than a
reflection. Moiré is born at the sensor, after the lens, so it stays sharp; generated
before stage 4 it would be washed out.

Three modelling decisions separate this from a generic filter stack.

**Moiré is generated as the beat, not as the two grids.** The effect arises from two
sampling grids that almost agree — say screen pixels every 0.100 mm, sampled by the
sensor every 0.104 mm. The alignment error accumulates and resets, producing a pattern
roughly 25 samples wide out of two grids 0.1 mm apart. The visible interference is far
coarser than either grid, and the grids themselves are finer than a single output
pixel, so the beat is synthesised directly. Its frequency is therefore measured in
cycles across the image width, which holds the appearance stable across resolutions;
below a two-pixel fringe period it stops describing the output and the generator warns.
Every other parameter carries its unit in its own name. Each attack draws its frequency
from `moire_min_cycles_per_width` to `moire_max_cycles_per_width` on a logarithmic scale,
and its angle from anywhere within `moire_max_angle_deg` of upright. A generator with a
single frequency and angle gave every attack the same fringe, and a judge trained on those
attacks learned that fringe and nothing else.

**The modulation is multiplicative.** Moiré modulates emitted light, so a black region,
where the panel emits nothing, has nothing to modulate and stays black. An additive
pattern would lift those blacks, which no real recapture does.

**The fringes are chromatic.** A display's red, green and blue subpixels sit at
different physical positions, so each channel aliases out of step with the others. A
per-channel phase offset (`moire_chroma_shift_rad`) reproduces the rainbow banding of a
real recapture; without it the fringes come out grey, which is the wrong artifact.

**The model is deliberately shallow.** An earlier draft summed three harmonics per
grating and modulated the whole pattern by a slow beat envelope. An ablation over the
individual terms, rendered at the 224 px working resolution, showed the cross grating
and the chroma offset carrying the visual change while the harmonics and the envelope
contributed almost nothing for four extra parameters and a nested loop, so both came
out. What survives is one sinusoid per grating direction. If the detector later shows
those terms matter, they come back with a measurement behind them.

Every stage switches off through its own parameter, so the neutral configuration is the
identity function — which is what makes an ablation over the individual artifacts
possible. Parameters are tuned visually against a synthetic test chart or a real face:

```bash
uv run python scripts/preview_moire.py --sweep saturation=0.8:1.6:5
```

The eye is not the final judge of a parameter. `calibrate_generator.py` measures, on the
training partition of each corpus, how the real replay and print crops differ from the
bona fide ones: saturation, contrast, black and white levels, sharpness, and the power
of the colour fringes in bands of cycles per width. It then measures the same on attacks
rendered from those bona fide crops. A generated attack is calibrated when its ratios
to bona fide match the real replay's. Test partitions never take part.

```bash
uv run python scripts/calibrate_generator.py
```

The defaults come from that report. Against bona fide, the real replays of both corpora
are more saturated (×1.15 to ×1.27) and sharper (×1.26 to ×1.32). Their contrast rises on
MSU-MFSD (×1.51) and barely moves on Replay-Mobile (×1.04). Most of their extra colour
power follows from those tone changes alone. What is left is an excess below 16 cycles
per width on MSU-MFSD, which is where the moiré belongs, and none on Replay-Mobile, which
replays on a matte screen. The first defaults did the opposite on every count: they
desaturated, flattened and blurred, and they concentrated their fringes at 20 cycles. The
calibrated defaults follow the MSU-MFSD replays closely across every band and land between
the two corpora on tone.

Hand-tuning against a report that covers both corpora leaks the test corpus into a
cross-dataset experiment. `--fit` therefore calibrates on one corpus alone: a random
search draws tone, blur, noise and fringe settings and keeps the one whose attacks sit
closest to that corpus's training replays. Tone and colour count equally in that
distance. The fringe band may move but keeps its width, because a single frequency can
match the mean statistics and is exactly the shortcut a judge learns. Each fit lives in
`experiments/`, and `train_judge.py --generator-config` trains on it, so a judge trained on
one corpus uses a generator that has only ever seen that corpus.

```bash
uv run python scripts/calibrate_generator.py --fit replay-mobile --trials 250 --output experiments/moire-fitted-to-replay-mobile.json
```

Once the parameters are settled, one attack is rendered for every bona fide image in a
folder. The run writes a `manifest.json` next to the attacks recording the exact generator
settings and the source each attack came from, so a figure in the thesis traces back to
the images and the parameters behind it.

```bash
uv run python scripts/generate_attacks.py --input data/msu/real --output data/msu/moire --limit 1000
```

### Evaluation

The reference judge is an ImageNet-pretrained ResNet fine-tuned on the real crops of
one corpus and tested on the test partitions of both, which gives the intra-dataset
and the cross-dataset figure of the same model. MSU-MFSD has no development
partition, so five of its fifteen training subjects are held out, whole and always
the same five, to fix the decision threshold; Replay-Mobile uses its own. The only
augmentation is a horizontal flip: colour jitter or blur would erase the very cues a
recapture leaves.

```bash
uv run python scripts/train_judge.py --train-corpus msu --output results/judges/msu-real-seed1
```

`--data` chooses the attacks the judge learns from, and is the variable the thesis
measures: `real` uses the corpus's own, `synthetic` replaces them with one moire
recapture rendered from each bona fide training crop, and `mixed` uses both. The
development partition stays real in all three, so only the training data changes
between them.

`explain_judge.py` shows where a trained judge looks. It computes Grad-CAM of the
attack logit over the test crops of both corpora, split by species, together with one
synthetic replay rendered from each bona fide test crop. It writes `attention.json`
next to the run: the share of attention that falls on the face box and the mean map.
With `--overlays N`, it also writes N heat-map examples per group. Grad-CAM keeps only
evidence for an attack, so a crop judged plainly genuine has an empty map; those are
counted rather than averaged in.

```bash
uv run python scripts/explain_judge.py --run results/judges/msu-real-seed1
```

Attacks are scored by a liveness detector and evaluated using standard PAD metrics:

- **APCER** — Attack Presentation Classification Error Rate (attacks accepted as genuine)
- **BPCER** — Bona Fide Presentation Classification Error Rate (genuine samples rejected)
- **ACER** — Average Classification Error Rate
- **EER** — Equal Error Rate
- **HTER** — Half Total Error Rate, at a threshold fixed on a development set; when a
  whole interval of thresholds equalises the two error rates there, its midpoint
- **AUC** — threshold-independent measure of separability

Metric definitions follow ISO/IEC 30107-3. Accuracy, precision, recall and F1, with the
attack as the positive class, are reported at the same development threshold. As a
sensitivity analysis, the thesis also reports the errors at a second threshold, set from
the development bona fide alone: the lowest one that rejects at most 5% of them.

### Reproducing the thesis

Every table and figure of the results chapter is rendered from `results/` by one
script, which also prints the paired seed-by-seed differences the text quotes, with
their bootstrap intervals:

```bash
uv run python scripts/render_thesis.py
```

It reads the ten-seed runs of `results/clean`, the first-generator runs of
`results/judges` and the calibration reports, and writes the `.tex` tables and `.pdf`
figures into `monographs/`. `ablate_generator.py` writes the `ablation.json` behind the
stage table: the median score a trained judge gives to attacks rendered with each
stage of the generator switched on or off.

```bash
uv run python scripts/ablate_generator.py --run results/clean/msu-synthetic-seed1
```

The first generator, tuned by eye, is no longer the code on `main`: its experiments
were run at the tag `first-generator`. The ablation and calibration scripts run there
unchanged, against that checkout's package:

```bash
git worktree add ../first-generator first-generator
uv run --project ../first-generator python scripts/ablate_generator.py --run results/judges/msu-synthetic-seed1
uv run --project ../first-generator python ../first-generator/scripts/calibrate_generator.py --output results/calibration-first-generator.json
```

## Datasets

Experiments use two face PAD datasets, MSU-MFSD and Replay-Mobile, as the source of
bona fide samples and as the real-attack reference. Both ship with per-frame face
annotations, so crops are cut from the official boxes rather than from a detector of
our own, which keeps the preprocessing comparable with published work. Each corpus
keeps its official partitions: MSU-MFSD splits 15 training and 20 test subjects,
Replay-Mobile 12 training, 16 development and 12 test subjects. Attacks are labelled
by species, replay or print, because the generators here only simulate replay and
ISO/IEC 30107-3 reports APCER per species.

```bash
uv run python scripts/extract_frames.py --corpus msu --root data/msu --output data/frames/msu
uv run python scripts/extract_frames.py --corpus replay-mobile --root data/replay-mobile --output data/frames/replay-mobile
```

Each run samples evenly spaced frames from every video, cuts a square crop centred on
the face and writes it losslessly as PNG, so that no compression artifact is mistaken
for a recapture cue. A `manifest.json` next to the crops records the settings and the
partition, subject, species, video and frame behind every crop.

> **Note:** dataset files are not distributed in this repository. They are subject to their original licenses and must be obtained from their respective providers.

## Repository structure

The code follows a thin **ports and adapters** layout. The domain layer owns the
two contracts of the experimental loop — `AttackGenerator` and `LivenessDetector`
— and depends on nothing but NumPy; every strategy and every detector is an
adapter behind one of those ports. That is what keeps the strategies in the table
above directly comparable, instead of three separate codebases.

```
synthetic-pad/
├── src/synthetic_pad/
│   ├── domain/        # pure types + the generator/detector ports (no I/O)
│   ├── generators/    # attack strategies
│   ├── judge/         # liveness detectors (a fine-tuned ResNet, with Grad-CAM)
│   ├── evaluation/    # metrics (APCER, BPCER, ACER, EER, HTER, AUC) and the per-species report
│   ├── dataset.py     # image I/O and batch attack generation
│   ├── calibration.py # image statistics that match generated attacks to real ones
│   ├── corpora.py     # MSU-MFSD and Replay-Mobile readers: partitions, attack species, face boxes
│   └── frames.py      # face crops sampled from the corpus videos, and their partitions
├── scripts/           # command line tools: tuning, generation, training, the thesis assets
├── experiments/       # generator settings fitted to one corpus each
├── tests/             # mirrors src/
├── data/              # datasets (gitignored)
└── results/           # experiment outputs (gitignored)
```

Two conventions carry most of the reproducibility weight. Parameters live in a
frozen config dataclass per generator, never as loose keyword arguments, so a run
is described by one serialisable object, which `describe()` returns — any figure
in the thesis traces back to the parameters that produced it. And generators own
an explicitly seeded `numpy.random.Generator` rather than calling the global
`numpy.random` functions.

Dependencies only ever point *into* `domain/`. Nothing in `generators/`, `judge/`
or `evaluation/` imports from a sibling, which is what lets a strategy be swapped
without touching the detector that scores it.

## Development

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock`, so an experiment can be re-run against the exact same environment.

```bash
uv sync                 # create the venv and install everything
uv run pre-commit install
```

The gates, which are exactly what the pipeline runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

`pre-commit` runs the fast ones on every commit, so a failure surfaces before the
push. It also runs `gitleaks`, which is the only check that can stop a secret
*before* it enters the history rather than reporting it afterwards.

`main` is protected: no direct pushes, every change lands through a pull request,
and both workflows must be green to merge. **Health check** runs `ruff`,
`mypy --strict` and `pytest`; **Security** runs `gitleaks` — over the pull request's commits on
a PR, and over the whole history on a manual `gh workflow run security.yml`.
Dependabot opens weekly dependency updates.

## Status

🚧 Work in progress — active development for TCC II (2026.2).

- [x] Project scaffolding and generator interface
- [x] CI/CD pipeline and security gates
- [x] Moiré / display recapture baseline
- [ ] Liveness detector (judge)
- [x] Evaluation metrics
- [ ] First end-to-end experiment
- [ ] Diffusion-based generation
- [ ] VLM-conditioned generation
- [ ] Comparative analysis

## Author

**Micael Angelo Sabadin Presotto**
Advisor: Prof. Elder Rizzon Santos, Dr.
Departamento de Informática e Estatística (INE) — CTC — UFSC
Florianópolis, 2026
