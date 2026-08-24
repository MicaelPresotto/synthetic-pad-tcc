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

### Evaluation

Attacks are scored by a liveness detector and evaluated using standard PAD metrics:

- **APCER** — Attack Presentation Classification Error Rate (attacks accepted as genuine)
- **BPCER** — Bona Fide Presentation Classification Error Rate (genuine samples rejected)
- **ACER** — Average Classification Error Rate
- **EER** — Equal Error Rate
- **AUC** — threshold-independent measure of separability

Metric definitions follow ISO/IEC 30107-3.

## Datasets

Experiments use established face PAD datasets (MSU-MFSD, Replay-Attack) as the source of bona fide samples and as the real-attack reference for comparison.

> **Note:** dataset files are not distributed in this repository. They are subject to their original licenses and must be obtained from their respective providers.

## Repository structure

```
synthetic-pad/
├── generators/      # attack generation strategies (shared interface)
├── judge/           # liveness detectors used to score attacks
├── experiments/     # experiment scripts and notebooks
├── evaluation/      # metrics (APCER, BPCER, ACER, EER, AUC)
├── data/            # datasets (gitignored)
└── results/         # experiment outputs
```

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
```

`pre-commit` runs the fast ones on every commit, so a failure surfaces before the
push. It also runs `gitleaks`, which is the only check that can stop a secret
*before* it enters the history rather than reporting it afterwards.

`main` is protected: no direct pushes, every change lands through a pull request,
and both workflows must be green to merge. **Health check** runs `ruff` and
`mypy --strict`; **Security** runs `gitleaks` — over the pull request's commits on
a PR, and over the whole history on a manual `gh workflow run security.yml`.
Dependabot opens weekly dependency updates.

## Status

🚧 Work in progress — active development for TCC II (2026.2).

- [ ] Project scaffolding and generator interface
- [x] CI/CD pipeline and security gates
- [ ] Moiré / display recapture baseline
- [ ] Liveness detector (judge)
- [ ] Evaluation metrics
- [ ] First end-to-end experiment
- [ ] Diffusion-based generation
- [ ] VLM-conditioned generation
- [ ] Comparative analysis

## Author

**Micael Angelo Sabadin Presotto**
Advisor: Prof. Elder Rizzon Santos, Dr.
Departamento de Informática e Estatística (INE) — CTC — UFSC
Florianópolis, 2026
