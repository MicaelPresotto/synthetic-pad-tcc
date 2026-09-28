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
| 2 | Tone | The panel cannot emit true black and the camera clips the highlights, so contrast and dynamic range both shrink | `contrast`, `black_level`, `white_level` |
| 3 | Glare | A light source in the room mirrored on the screen glass | `glare_intensity`, `glare_extent` (fraction of the image width) |
| 4 | Lens | Defocus and the finite pixel fill factor soften everything | `blur_sigma_px` |
| 5 | Moiré | The screen's pixel grid beating against the sensor's | `moire_amplitude`, `moire_cycles_per_width`, `moire_angle_deg`, `moire_chroma_shift_rad`, `moire_cross_weight` |
| 6 | Noise | Sensor read noise on a dim, indirect subject | `noise_sigma_levels` |

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
pixel, so the beat is synthesised directly. `moire_cycles_per_width` is therefore measured in
cycles across the image width, which holds the appearance stable across resolutions;
below a two-pixel fringe period the parameter stops describing the output and the
generator warns. Every other parameter carries its unit in its own name. The default of 20 cycles — an 11 px fringe at the 224 px working
resolution — was picked from the sweep below, and the ±25% per-image jitter keeps the
whole range inside the band that still reads as moiré rather than as fine dither.

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
uv run python scripts/preview_moire.py --sweep moire_cycles_per_width=8:70:8
```

Once the parameters are settled, one attack is rendered for every bona fide image in a
folder. The run writes a `manifest.json` next to the attacks recording the exact generator
settings and the source each attack came from, so a figure in the thesis traces back to
the images and the parameters behind it.

```bash
uv run python scripts/generate_attacks.py --input data/msu/real --output data/msu/moire --limit 1000
```

### Evaluation

Attacks are scored by a liveness detector and evaluated using standard PAD metrics:

- **APCER** — Attack Presentation Classification Error Rate (attacks accepted as genuine)
- **BPCER** — Bona Fide Presentation Classification Error Rate (genuine samples rejected)
- **ACER** — Average Classification Error Rate
- **EER** — Equal Error Rate
- **HTER** — Half Total Error Rate, at a threshold fixed on a development set
- **AUC** — threshold-independent measure of separability

Metric definitions follow ISO/IEC 30107-3.

## Datasets

Experiments use established face PAD datasets (MSU-MFSD, Replay-Attack) as the source of bona fide samples and as the real-attack reference for comparison.

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
│   ├── judge/         # liveness detectors
│   ├── evaluation/    # metrics (APCER, BPCER, ACER, EER, HTER, AUC)
│   └── dataset.py     # image I/O and batch attack generation
├── scripts/           # command line tools: visual tuning, batch generation
├── experiments/       # application layer: wires a generator to a judge
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
