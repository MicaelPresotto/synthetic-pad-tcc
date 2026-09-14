# Working in this repository

Research code for a UFSC undergraduate thesis (TCC) on synthetic presentation
attacks for face liveness detection. It is an academic deliverable: the git
history and the code are read by an examining board, so both are written to be
read by a person, not only executed.

Read `README.md` first — it carries the research question, the moiré physics and
the repository layout. This file only states the rules an agent cannot infer
from the code.

## The one architectural constraint

`domain/` owns the vocabulary — `AttackGenerator`, `LivenessDetector`,
`RgbImage`, the validators — and depends on nothing but NumPy.
`generators/`, `judge/` and `evaluation/` may import `domain/` and never each
other. Dependencies point inward, always. That is the whole of the ports and
adapters layout here, and it is deliberately thin.

## Design principles, in order of authority

KISS > DRY > DDD > clean code.

- **KISS.** Build for today's need. Name the second caller before extracting an
  abstraction; if there isn't one, don't extract. No language feature that saves
  no work — PEP 695 generics were rejected for this reason, PEP 695 `type`
  aliases were kept because they read plainly.
- **DRY.** A rule is written once. Repeated validation lives in
  `domain/validation.py`; the preview CLI generates its flags from
  `dataclasses.fields(MoireConfig)` instead of listing them.
- **DDD.** The layering above, nothing more.
- **Clean code.** Names carry the meaning. Explicitly rejected from the book:
  "every function is 5 lines", "one level of abstraction per function",
  "extract till you drop" — they manufacture indirection.

## Conventions

- **No inline comments, no routine docstrings.** A comment means the naming
  failed: rename the variable or extract a named function. Magic numbers become
  named module constants. Keep a one-line docstring only for what a typed
  signature cannot express — a unit or value range, an acronym, a standard being
  followed, a sign convention. Never an `Args:`/`Returns:` block.
- **Never `from __future__ import annotations`.** It turns `field.type` into a
  string and breaks the runtime introspection the preview CLI depends on. `ruff`
  TID251 enforces this.
- **Generator parameters live in a frozen config dataclass**, validated in
  `__post_init__`, never as loose keyword arguments. `describe()` returns it, so
  every figure in the thesis traces back to the parameters that produced it.
- **Randomness is an explicitly seeded `numpy.random.Generator`** owned by the
  generator. Never the global `numpy.random` functions.
- **No new `.md` files.** Documentation goes in `README.md`. Every file,
  dependency and config entry must be demonstrably in use.
- Code, commits, PRs and documentation are written in English; the conversation
  with the author is in Portuguese.

## Gates

Run all four green before committing. They are exactly what CI runs.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

`pytest` runs with `filterwarnings = ["error"]`, so a warning fails the suite —
assert on it with `pytest.warns` when it is the behaviour under test.

## Git

- `main` is protected. Every change lands through a pull request, squash-merged.
- **Never list an AI as an author.** No `Co-Authored-By` trailer, no attribution
  anywhere in the history.
- **Never force-push.** Changes to an open PR go in as ordinary follow-up
  commits, even when they undo earlier work in the same branch.
- One PR, one purpose. A CI or tooling PR contains zero code changes.
- PR bodies are a title plus a few short paragraphs — no checklists, no gate
  tables.

## What never enters git

`data/` and `results/` are gitignored. Biometric datasets are licensed by their
providers and are never committed, redistributed, or sent to a third-party
service — not a sample, not a crop, not a thumbnail.
