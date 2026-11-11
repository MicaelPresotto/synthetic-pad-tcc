import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter

from synthetic_pad.corpora import (
    BONA_FIDE,
    DEVELOPMENT,
    MSU,
    PRINT,
    REPLAY,
    REPLAY_MOBILE,
    TEST,
    TRAIN,
)
from synthetic_pad.evaluation.metrics import (
    acer,
    apcer,
    bpcer,
    bpcer_threshold,
    classification_metrics,
)
from synthetic_pad.frames import FRAMES_PER_VIDEO, CropSample, load_partitions, select
from synthetic_pad.video_scores import SYNTHETIC_REPLAYS

CORPUS_NAMES = {MSU: "MSU-MFSD", REPLAY_MOBILE: "Replay-Mobile"}
DIRECTIONS = ((MSU, REPLAY_MOBILE), (REPLAY_MOBILE, MSU))
CONFIGURATIONS = ("real", "synthetic", "mixed")
SPECIES = ("replay", "print")
MOTION_CUES = ("blinks", "pulse")
RECORDED_KINDS = (PRINT, "replay of a photo", "replay of a video")
STAGES = ("geometry", "tone", "glare", "lens", "moire", "noise")
TONE_STATISTICS = ("saturation", "contrast", "black_level", "white_level", "sharpness")
GENERATED = "generated"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CONFIDENCE = 0.95
TARGET_BPCER = 0.05
FIGURE_WIDTH_IN = 6.3
FIGURE_HEIGHT_IN = 2.6
SPECTRUM_TICKS = (0.5, 1.0, 2.0, 5.0, 10.0)
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#52514e"
GRID = "#d9d8d4"
SOURCE_LINE = r"{\footnotesize Source: prepared by the author (2026).}"
NOT_RUN = "--"
COLUMN_GAP_PT = 4
MSU_SPLIT_DRAWS = 8
SCORE_EPSILON = 1e-7
LOG_ODDS_BIN_WIDTH = 0.5
BAND_OPACITY = 0.15
HISTOGRAM_OPACITY = 0.6
SYNTHETIC_PER_BONA_FIDE = 1
PARTITION_NAMES = {TRAIN: "Training", DEVELOPMENT: "Development", TEST: "Test"}
CENTRED_X = ">{\\centering\\arraybackslash}X"

type Report = dict[str, Any]
type Runs = dict[tuple[str, str], dict[int, Path]]


def find_runs(results: Path) -> Runs:
    """Run folders by (source corpus, training data), keyed by seed so that runs pair up."""
    found: Runs = defaultdict(dict)
    for path in sorted(results.glob("*/report.json")):
        training = json.loads(path.read_text())["training"]
        found[(training["corpus"], training["data"])][training["config"]["seed"]] = path.parent
    return found


def report(run: Path) -> Report:
    return dict(json.loads((run / "report.json").read_text()))


def measured(runs: dict[int, Path], pick: Callable[[Report], float]) -> np.ndarray:
    return np.array([pick(report(run)) for _, run in sorted(runs.items())])


def arrow(source: str, target: str) -> str:
    return f"{CORPUS_NAMES[source]} $\\rightarrow$ {CORPUS_NAMES[target]}"


def percent(values: np.ndarray) -> str:
    return f"{100 * values.mean():.1f} $\\pm$ {100 * values.std():.1f}"


def ratio(values: np.ndarray) -> str:
    return f"{values.mean():.3f} $\\pm$ {values.std():.3f}"


def cross_auc(species: str | None) -> Callable[[Report], float]:
    if species is None:
        return lambda r: float(r["cross"]["all_attacks"]["auc"])
    return lambda r: float(r["cross"]["by_species"][species]["auc"])


def intra_auc(species: str | None) -> Callable[[Report], float]:
    if species is None:
        return lambda r: float(r["intra"]["all_attacks"]["auc"])
    return lambda r: float(r["intra"]["by_species"][species]["auc"])


def hter_of(scenario: str) -> Callable[[Report], float]:
    return lambda r: float(r[scenario]["all_attacks"]["hter"])


def table(caption: str, label: str, columns: str, header: str, rows: Sequence[str]) -> str:
    body = "\n".join(rows)
    return (
        "\\begin{center}\n"
        "\\begin{minipage}{\\textwidth}\n"
        "\t\\centering\n"
        f"\t\\captionof{{table}}{{{caption}}}\n"
        f"\t\\label{{{label}}}\n"
        "\t\\footnotesize\n"
        "\t\\renewcommand{\\arraystretch}{1.2}\n"
        f"\t\\setlength{{\\tabcolsep}}{{{COLUMN_GAP_PT}pt}}\n"
        f"\t\\begin{{tabularx}}{{\\textwidth}}{{{columns}}}\n"
        "\t\t\\toprule\n"
        f"\t\t{header} \\\\\n"
        "\t\t\\midrule\n"
        f"{body}\n"
        "\t\t\\bottomrule\n"
        "\t\\end{tabularx}\n\n"
        f"\t{SOURCE_LINE}\n"
        "\\end{minipage}\n"
        "\\end{center}\n"
    )


def thousands(count: int) -> str:
    return f"{count:,}".replace(",", "{,}")


def composition_rows(
    samples: Sequence[CropSample], partition: str
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Bona fide, print, replay and synthetic crops per set; synthetic ones enter training only."""
    species = Counter(sample.species for sample in select(samples, partition))
    bona_fide, printed, replayed = species[BONA_FIDE], species[PRINT], species[REPLAY]
    name = PARTITION_NAMES[partition]
    if partition != TRAIN:
        return [(name, (bona_fide, printed, replayed, 0))]
    synthetic = SYNTHETIC_PER_BONA_FIDE * bona_fide
    return [
        (f"{name}, real", (bona_fide, printed, replayed, 0)),
        (f"{name}, synthetic", (bona_fide, 0, 0, synthetic)),
        (f"{name}, mixed", (bona_fide, printed, replayed, synthetic)),
    ]


def composition_table(partitions: dict[str, list[CropSample]]) -> str:
    rows = []
    for corpus, samples in partitions.items():
        rows.append(f"\t\t\\multicolumn{{7}}{{l}}{{\\textit{{{CORPUS_NAMES[corpus]}}}}} \\\\")
        for partition in PARTITION_NAMES:
            for label, (bona_fide, printed, replayed, synthetic) in composition_rows(
                samples, partition
            ):
                attacks = printed + replayed + synthetic
                cells = [
                    label,
                    *(thousands(count) for count in (bona_fide, printed, replayed, synthetic)),
                    thousands(bona_fide + attacks),
                    f"{100 * synthetic / attacks:.0f}",
                ]
                rows.append("\t\t" + " & ".join(cells) + " \\\\")
        if corpus != list(partitions)[-1]:
            rows.append("\t\t\\midrule")
    return table(
        f"Number of face crops in each partition, {FRAMES_PER_VIDEO} per video. The training set "
        "of each configuration holds as many synthetic replay attacks as bona fide training "
        "crops, and the development and test partitions hold the same real crops in "
        "every configuration. "
        "The last column is the share of the attacks that are synthetic (\\%).",
        "tab:composition",
        "Xcccccc",
        "\\textbf{Set} & \\textbf{Bona fide} & \\textbf{Print} & \\textbf{Replay} & "
        "\\textbf{Synthetic} & \\textbf{Total} & \\textbf{Synthetic (\\%)}",
        rows,
    )


def direction_heading(source: str, target: str, columns: int) -> str:
    return f"\t\t\\multicolumn{{{columns}}}{{l}}{{\\textit{{{arrow(source, target)}}}}} \\\\"


def baseline_table(runs: Runs) -> str:
    rows = []
    for source, target in DIRECTIONS:
        real = runs[(source, "real")]
        scenarios = ((source, "intra", intra_auc), (target, "cross", cross_auc))
        for tested, scenario, auc_of in scenarios:
            cells = [
                arrow(source, tested),
                percent(measured(real, hter_of(scenario))),
                *(ratio(measured(real, auc_of(species))) for species in (None, *SPECIES)),
            ]
            rows.append("\t\t" + " & ".join(cells) + " \\\\")
    return table(
        "Detector trained on real data only: HTER (\\%) at the development threshold and AUC, "
        "mean and standard deviation over ten seeds.",
        "tab:baseline",
        "Xcccc",
        "\\textbf{Train $\\rightarrow$ test} & \\textbf{HTER} & \\textbf{AUC} & "
        "\\textbf{AUC replay} & \\textbf{AUC print}",
        rows,
    )


def main_table(runs: Runs) -> str:
    rows = []
    for source, target in DIRECTIONS:
        rows.append(direction_heading(source, target, 6))
        for data in CONFIGURATIONS:
            trained = runs[(source, data)]
            cells = [
                data.capitalize(),
                percent(measured(trained, hter_of("intra"))),
                percent(measured(trained, hter_of("cross"))),
                *(ratio(measured(trained, cross_auc(species))) for species in (None, *SPECIES)),
            ]
            rows.append("\t\t" + " & ".join(cells) + " \\\\")
        if source != DIRECTIONS[-1][0]:
            rows.append("\t\t\\midrule")
    return table(
        "Detectors trained on real, synthetic and mixed attacks, with the generator calibrated on "
        "the source dataset alone: intra-dataset HTER (\\%), and cross-dataset HTER (\\%) and AUC, "
        "mean and standard deviation over ten seeds.",
        "tab:main",
        "Xccccc",
        "\\textbf{Attacks} & \\textbf{HTER intra} & "
        "\\textbf{HTER cross} & \\textbf{AUC cross} & \\textbf{AUC replay cross} & "
        "\\textbf{AUC print cross}",
        rows,
    )


def partition_scores(run: Path, partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attack scores, bona fide scores and the species of each attack."""
    scores = json.loads((run / "scores.json").read_text())[partition]
    attacks = [s for s in scores if s["species"] != BONA_FIDE]
    return (
        np.array([s["score"] for s in attacks]),
        np.array([s["score"] for s in scores if s["species"] == BONA_FIDE]),
        np.array([s["species"] for s in attacks]),
    )


def scenario_metrics(run: Path, scenario: str) -> dict[str, float]:
    """PAD and general classification metrics of one test partition, at the threshold the
    development partition fixed."""
    section = report(run)[scenario]
    attacks, bona_fide, _ = partition_scores(run, scenario)
    return {
        **{f"apcer {species}": section["by_species"][species]["apcer"] for species in SPECIES},
        "bpcer": section["bpcer"],
        **classification_metrics(attacks, bona_fide, section["threshold"]),
    }


def metrics_table(runs: Runs) -> str:
    columns = ("apcer replay", "apcer print", "bpcer", "accuracy", "precision", "recall", "f1")
    rows = []
    for source, target in DIRECTIONS:
        rows.append(direction_heading(source, target, len(columns) + 2))
        for data in CONFIGURATIONS:
            for scenario in ("intra", "cross"):
                trained = runs[(source, data)].values()
                per_seed = [scenario_metrics(run, scenario) for run in trained]
                means = [100 * np.mean([m[column] for m in per_seed]) for column in columns]
                cells = [
                    data.capitalize() if scenario == "intra" else "",
                    scenario.capitalize(),
                    *(f"{value:.1f}" for value in means),
                ]
                rows.append("\t\t" + " & ".join(cells) + " \\\\")
        if source != DIRECTIONS[-1][0]:
            rows.append("\t\t\\midrule")
    return table(
        "PAD and general classification metrics (\\%) at the development threshold, with the "
        "attack as the positive class. Mean over ten seeds.",
        "tab:metrics",
        "ll" + CENTRED_X * len(columns),
        "\\textbf{Attacks} & \\textbf{Test} & \\textbf{APCER replay} & "
        "\\textbf{APCER print} & \\textbf{BPCER} & \\textbf{Accuracy} & \\textbf{Precision} & "
        "\\textbf{Recall} & \\textbf{F1}",
        rows,
    )


def bona_fide_rule(run: Path, scenario: str) -> dict[str, float]:
    """Errors of one test partition at the threshold that keeps the development BPCER at or below
    TARGET_BPCER, set from the development bona fide alone."""
    _, development_bona_fide, _ = partition_scores(run, "development")
    threshold = bpcer_threshold(development_bona_fide, TARGET_BPCER)
    attacks, bona_fide, species = partition_scores(run, scenario)
    return {
        "hter": acer(attacks, bona_fide, threshold),
        **{f"apcer {kind}": apcer(attacks[species == kind], threshold) for kind in SPECIES},
        "bpcer": bpcer(bona_fide, threshold),
    }


def threshold_table(runs: Runs) -> str:
    rows = []
    for source, target in DIRECTIONS:
        rows.append(direction_heading(source, target, 8))
        for data in CONFIGURATIONS:
            trained = list(runs[(source, data)].values())
            cross = [bona_fide_rule(run, "cross") for run in trained]
            values = [
                np.mean([hter_of("intra")(report(run)) for run in trained]),
                np.mean([bona_fide_rule(run, "intra")["hter"] for run in trained]),
                np.mean([hter_of("cross")(report(run)) for run in trained]),
                *(
                    np.mean([errors[column] for errors in cross])
                    for column in ("hter", "apcer replay", "apcer print", "bpcer")
                ),
            ]
            cells = [data.capitalize(), *(f"{100 * value:.1f}" for value in values)]
            rows.append("\t\t" + " & ".join(cells) + " \\\\")
        if source != DIRECTIONS[-1][0]:
            rows.append("\t\t\\midrule")
    target = f"{100 * TARGET_BPCER:.0f}\\%"
    return table(
        "Error rates (\\%) under two threshold rules: the equal-error (EER) threshold of the "
        "development partition, and the threshold that rejects at most "
        f"{target} of the development bona fide crops, set without looking at any attack. "
        "Mean over ten seeds.",
        "tab:threshold",
        "l" + CENTRED_X * 7,
        "& \\multicolumn{2}{c}{\\textbf{HTER intra}} & "
        "\\multicolumn{2}{c}{\\textbf{HTER cross}} & "
        f"\\multicolumn{{3}}{{c}}{{\\textbf{{Cross at BPCER {target}}}}} \\\\\n"
        "\t\t\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-8}\n"
        "\t\t\\textbf{Attacks} & "
        f"\\textbf{{EER}} & \\textbf{{BPCER {target}}} & "
        f"\\textbf{{EER}} & \\textbf{{BPCER {target}}} & "
        "\\textbf{APCER replay} & \\textbf{APCER print} & \\textbf{BPCER}",
        rows,
    )


def percent_mean(runs: dict[int, Path], pick: Callable[[Report], float]) -> str:
    return f"{100 * measured(runs, pick).mean():.1f}"


def noise_table(noise_on: Runs, noise_off: Runs, noise_on_real_crops: Runs) -> str:
    """Cross-dataset errors with the noise stage of the generator on, off, and applied to every
    real training crop as well; the real configuration uses no generator when the stage is off."""
    rows = []
    for source, target in DIRECTIONS:
        rows.append(direction_heading(source, target, 7))
        for data in CONFIGURATIONS:
            key = (source, data)
            hters = [
                percent_mean(runs[key], hter_of("cross")) if key in runs else NOT_RUN
                for runs in (noise_on, noise_off, noise_on_real_crops)
            ]
            neutral = noise_on_real_crops[key].values()
            cells = [
                data.capitalize(),
                *hters,
                *(
                    f"{measured(runs[key], cross_auc('replay')).mean():.3f}"
                    for runs in (noise_on, noise_on_real_crops)
                ),
                f"{100 * np.mean([bona_fide_rule(run, 'cross')['hter'] for run in neutral]):.1f}",
            ]
            rows.append("\t\t" + " & ".join(cells) + " \\\\")
        if source != DIRECTIONS[-1][0]:
            rows.append("\t\t\\midrule")
    target = f"{100 * TARGET_BPCER:.0f}\\%"
    return table(
        "Cross-dataset HTER (\\%) and replay AUC with the noise stage of the generator on, as in "
        "Table~\\ref{tab:main}, switched off, and applied to every real training crop as well, so "
        "that the noise level no longer tells the classes apart; the last column repeats the "
        f"condition with noise on every real crop, evaluated at the BPCER {target} threshold of "
        "Table~\\ref{tab:threshold}, and the replay AUC is given for the first and the third "
        "condition only. Mean over ten seeds.",
        "tab:noise",
        "l" + CENTRED_X * 6,
        "& \\multicolumn{3}{c}{\\textbf{HTER cross}} & "
        "\\multicolumn{2}{c}{\\textbf{AUC replay cross}} & \\textbf{HTER cross} \\\\\n"
        "\t\t\\cmidrule(lr){2-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-7}\n"
        "\t\t\\textbf{Attacks} & \\textbf{Noise on} & "
        "\\textbf{Noise off} & \\textbf{On real crops} & \\textbf{Noise on} & "
        f"\\textbf{{On real crops}} & \\textbf{{BPCER {target}}}",
        rows,
    )


def score_range(values: Sequence[float]) -> str:
    low, high = f"{min(values):.2f}", f"{max(values):.2f}"
    return low if low == high else f"{low}--{high}"


def first_generator_table(runs: Runs) -> str:
    rows = []
    for source, target in DIRECTIONS:
        rows.append(direction_heading(source, target, 5))
        for data in ("real", "synthetic"):
            trained = runs[(source, data)]
            cells = [
                data.capitalize(),
                percent(measured(trained, hter_of("intra"))),
                ratio(measured(trained, intra_auc(None))),
                percent(measured(trained, hter_of("cross"))),
                ratio(measured(trained, cross_auc(None))),
            ]
            rows.append("\t\t" + " & ".join(cells) + " \\\\")
        if source != DIRECTIONS[-1][0]:
            rows.append("\t\t\\midrule")
    return table(
        "Detectors trained on real attacks and on the synthetic attacks of the first generator "
        "alone: HTER (\\%) at the development threshold and AUC, within the source dataset and "
        "across datasets, mean and standard deviation over three seeds.",
        "tab:first-generator",
        "Xcccc",
        "\\textbf{Attacks} & \\textbf{HTER intra} & \\textbf{AUC intra} & "
        "\\textbf{HTER cross} & \\textbf{AUC cross}",
        rows,
    )


def ablation_table(runs: Runs) -> str:
    groups = [(data, corpus) for data in ("synthetic", "real") for corpus in CORPUS_NAMES]
    medians = {
        group: [
            json.loads((run / "ablation.json").read_text())["median_score"]
            for run in runs[(group[1], group[0])].values()
        ]
        for group in groups
    }
    others = [stage for stage in STAGES if stage != "moire"]
    lines = (
        ("Every stage", ["every stage"]),
        ("Every stage but moiré", ["every stage but moire"]),
        ("Moiré alone", ["moire alone"]),
        ("Any other stage alone", [f"{stage} alone" for stage in others]),
    )
    rows = []
    for title, variants in lines:
        cells = [
            score_range([seed[variant] for seed in medians[group] for variant in variants])
            for group in groups
        ]
        rows.append("\t\t" + " & ".join([title, *cells]) + " \\\\")
    return table(
        "Median score of trained detectors on attacks rendered by the first generator from the "
        "bona fide test crops of their own dataset, with one stage switched on or off, over "
        "three seeds; the ranges span the seeds.",
        "tab:ablation-v1",
        "Xcccc",
        "& \\multicolumn{2}{c}{\\textbf{Trained on synthetic attacks}} & "
        "\\multicolumn{2}{c}{\\textbf{Trained on real attacks}} \\\\\n"
        "\t\t\\textbf{Attack rendered with} & \\textbf{MSU-MFSD} & \\textbf{Replay-Mobile} & "
        "\\textbf{MSU-MFSD} & \\textbf{Replay-Mobile}",
        rows,
    )


def median_and_range(values: Sequence[float]) -> str:
    return f"{np.median(values):.2f} ({score_range(values)})"


def ablation_cell(runs: Runs, corpus: str, variant: str) -> str:
    return median_and_range(
        [
            json.loads((run / "ablation.json").read_text())["median_score"][variant]
            for run in runs[(corpus, "synthetic")].values()
        ]
    )


def noise_ablation_table(noise_on: Runs, noise_on_real_crops: Runs) -> str:
    groups = [(runs, corpus) for runs in (noise_on, noise_on_real_crops) for corpus in CORPUS_NAMES]
    lines = (
        ("Every stage", "every stage"),
        ("Every stage but noise", "every stage but noise"),
        ("Noise alone", "noise alone"),
        ("Every stage but moiré", "every stage but moire"),
    )
    rows = []
    for title, variant in lines:
        cells = [ablation_cell(runs, corpus, variant) for runs, corpus in groups]
        rows.append("\t\t" + " & ".join([title, *cells]) + " \\\\")
    return table(
        "Median score of the calibrated synthetic detectors on attacks rendered from the bona fide "
        "test crops of their own dataset, with one stage switched on or off. Median over ten "
        "seeds, with the range across seeds.",
        "tab:ablation-noise",
        "Xcccc",
        "& \\multicolumn{2}{c}{\\textbf{Noise on synthetic attacks}} & "
        "\\multicolumn{2}{c}{\\textbf{Noise on real crops too}} \\\\\n"
        "\t\t\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
        "\t\t\\textbf{Attack rendered with} & \\textbf{MSU-MFSD} & \\textbf{Replay-Mobile} & "
        "\\textbf{MSU-MFSD} & \\textbf{Replay-Mobile}",
        rows,
    )


def generators_table(real: Runs, variants: Sequence[tuple[str, Runs]]) -> str:
    def errors(runs: dict[int, Path]) -> list[str]:
        replay_auc = measured(runs, cross_auc("replay")).mean()
        return [percent_mean(runs, hter_of("cross")), f"{replay_auc:.3f}"]

    rows = []
    for source, target in DIRECTIONS:
        rows.append(direction_heading(source, target, 5))
        hter, replay_auc = errors(real[(source, "real")])
        reference = f"\\multicolumn{{4}}{{c}}{{HTER {hter}\\%, replay AUC {replay_auc}}}"
        rows.append(f"\t\tReal attacks only & {reference} \\\\")
        for title, runs in variants:
            cells = [errors(runs[(source, data)]) for data in ("synthetic", "mixed")]
            rows.append("\t\t" + " & ".join([title, *cells[0], *cells[1]]) + " \\\\")
        if source != DIRECTIONS[-1][0]:
            rows.append("\t\t\\midrule")
    return table(
        "Cross-dataset HTER (\\%) and replay AUC for each version of the generator, with no "
        "noise on the real training crops: the sinusoidal fringes of Table~\\ref{tab:main} with "
        "the fixed noise level of the calibration or with the noise matched to the real replays, "
        "the simulated screen with matched noise, rendering one or three synthetic attacks per "
        "bona fide crop, and the transfer of the amplitude spectrum of the real replays of the "
        "source dataset onto its bona fide crops, up to 0.07 or 0.02 cycles per pixel. Mean over "
        "ten seeds.",
        "tab:generators",
        "l" + CENTRED_X * 4,
        "& \\multicolumn{2}{c}{\\textbf{Synthetic}} & \\multicolumn{2}{c}{\\textbf{Mixed}} \\\\\n"
        "\t\t\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
        "\t\t\\textbf{Generator} & \\textbf{HTER} & \\textbf{AUC replay} & \\textbf{HTER} & "
        "\\textbf{AUC replay}",
        rows,
    )


def generators_ablation_table(sinusoidal: Runs, screen: Runs) -> str:
    groups = [
        (runs, corpus, moire)
        for runs, moire in ((sinusoidal, "moire"), (screen, "screen"))
        for corpus in CORPUS_NAMES
    ]
    lines = (
        ("Every stage", "every stage"),
        ("Every stage but noise", "every stage but noise"),
        ("Noise alone", "noise alone"),
        ("Every stage but its moiré", "every stage but {moire}"),
        ("Its moiré alone", "{moire} alone"),
    )
    rows = []
    for title, variant in lines:
        cells = [
            ablation_cell(runs, corpus, variant.format(moire=moire))
            for runs, corpus, moire in groups
        ]
        rows.append("\t\t" + " & ".join([title, *cells]) + " \\\\")
    return table(
        "Median score of the synthetic detectors trained with matched noise on attacks rendered "
        "from the bona fide test crops of their own dataset, with one stage switched on or off; "
        "the moiré is the sinusoidal fringes or the simulated screen. Median over ten seeds, with "
        "the range across seeds.",
        "tab:ablation-generators",
        "l" + CENTRED_X * 4,
        "& \\multicolumn{2}{c}{\\textbf{Sinusoidal fringes}} & "
        "\\multicolumn{2}{c}{\\textbf{Simulated screen}} \\\\\n"
        "\t\t\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
        "\t\t\\textbf{Attack rendered with} & \\textbf{MSU-MFSD} & \\textbf{Replay-Mobile} & "
        "\\textbf{MSU-MFSD} & \\textbf{Replay-Mobile}",
        rows,
    )


def architectures_table(architectures: Sequence[tuple[str, Runs]]) -> str:
    def errors(runs: dict[int, Path]) -> list[str]:
        replay_auc = measured(runs, cross_auc("replay")).mean()
        return [
            percent_mean(runs, hter_of("intra")),
            percent_mean(runs, hter_of("cross")),
            f"{replay_auc:.3f}",
        ]

    width = 1 + 3 * len(architectures)
    rows = []
    for source, target in DIRECTIONS:
        rows.append(direction_heading(source, target, width))
        for data in CONFIGURATIONS:
            cells = [cell for _, runs in architectures for cell in errors(runs[(source, data)])]
            rows.append("\t\t" + " & ".join([data.capitalize(), *cells]) + " \\\\")
        if source != DIRECTIONS[-1][0]:
            rows.append("\t\t\\midrule")
    groups = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\textbf{{{title}}}}}" for title, _ in architectures
    )
    rules = " ".join(
        f"\\cmidrule(lr){{{2 + 3 * index}-{4 + 3 * index}}}" for index in range(len(architectures))
    )
    measures = " & ".join(
        ["\\textbf{HTER intra} & \\textbf{HTER cross} & \\textbf{AUC replay}"] * len(architectures)
    )
    names = ", ".join(f"the {title}" for title, _ in architectures[:-1])
    names += f" and the {architectures[-1][0]}"
    return table(
        f"Intra-dataset HTER (\\%), cross-dataset HTER (\\%) and replay AUC of {names} "
        "trained on real, synthetic and mixed attacks, with the sinusoidal fringes and matched "
        "noise. Mean over ten seeds.",
        "tab:architectures",
        "l" + CENTRED_X * (width - 1),
        f"& {groups} \\\\\n\t\t{rules}\n\t\t\\textbf{{Attacks}} & {measures}",
        rows,
    )


def development_subjects(runs: dict[int, Path]) -> str:
    (subjects,) = {
        tuple(report(run)["training"]["msu_development_subjects"]) for run in runs.values()
    }
    return ", ".join(subjects)


def cross_hter_differences(trained: dict[int, Path], reference: dict[int, Path]) -> np.ndarray:
    """Cross-dataset HTER of each trained run minus its reference of the same seed, in points."""
    seeds = sorted(set(reference) & set(trained))
    pick = on_report(hter_of("cross"))
    return 100.0 * np.array([pick(trained[s]) - pick(reference[s]) for s in seeds])


def difference_with_interval(differences: np.ndarray) -> str:
    low, high = bootstrap_interval(differences)
    return f"${signed(differences.mean())}$ [${signed(low)}$, ${signed(high)}$]"


def signed(value: float) -> str:
    """One decimal with its sign; adding 0.0 turns a rounded -0.0 into +0.0."""
    return f"{round(value, 1) + 0.0:+.1f}"


def paired_difference(trained: dict[int, Path], reference: dict[int, Path]) -> tuple[str, str]:
    """The mean cross HTER difference with its bootstrap interval, and the seeds improved."""
    differences = cross_hter_differences(trained, reference)
    improved = int(np.sum(differences < 0))
    return difference_with_interval(differences), f"{improved}/{differences.size}"


def print_variant_differences(title: str, variant: Runs, reference: Runs) -> None:
    """Each training configuration of a variant against the same configuration of a reference
    detector, seed by seed."""
    print(f"\n== {title}")
    for source, target in DIRECTIONS:
        for data in CONFIGURATIONS:
            difference, improved = paired_difference(
                variant[(source, data)], reference[(source, data)]
            )
            print(
                f"  {CORPUS_NAMES[source]} -> {CORPUS_NAMES[target]}, {data:9s} cross HTER "
                f"{difference}  improved in {improved}"
            )


def mixstyle_table(real: Runs, mixed: Runs, mixstyle: Runs) -> str:
    rows = []
    for source, target in DIRECTIONS:
        rows.append(direction_heading(source, target, 6))
        reference = real[(source, "real")]
        for title, runs in (
            ("Real", reference),
            ("Real + MixStyle", mixstyle[(source, "real")]),
            ("Mixed", mixed[(source, "mixed")]),
            ("Mixed + MixStyle", mixstyle[(source, "mixed")]),
        ):
            difference = (
                (NOT_RUN, NOT_RUN) if runs is reference else paired_difference(runs, reference)
            )
            cells = [
                title,
                percent_mean(runs, hter_of("intra")),
                percent_mean(runs, hter_of("cross")),
                *difference,
                f"{measured(runs, cross_auc('replay')).mean():.3f}",
            ]
            rows.append("\t\t" + " & ".join(cells) + " \\\\")
        if source != DIRECTIONS[-1][0]:
            rows.append("\t\t\\midrule")
    return table(
        "Detectors trained with and without MixStyle on real or mixed attacks, with the "
        "sinusoidal fringes and matched noise: intra-dataset and cross-dataset HTER (\\%), "
        "the paired cross-dataset difference to the real detector without MixStyle with its "
        "95\\% bootstrap interval and the seeds in which it improved, and the replay AUC. "
        "A negative difference is a lower error than that of the real detector. "
        "Mean over ten seeds.",
        "tab:mixstyle",
        "l" + CENTRED_X * 2 + "cc" + CENTRED_X,
        "\\textbf{Attacks} & \\textbf{HTER intra} & \\textbf{HTER cross} & "
        "\\textbf{Cross $-$ real} & \\textbf{Seeds} & \\textbf{AUC replay}",
        rows,
    )


def depth_table(one_label: Runs, variants: Sequence[tuple[str, Runs]]) -> str:
    """One table per direction; the second refers to the first for its columns."""
    first_label = f"tab:depth-{DIRECTIONS[0][0]}"
    tables = []
    for source, target in DIRECTIONS:
        rows = []
        for data in CONFIGURATIONS:
            for title, runs in ((data.capitalize(), one_label), *variants):
                trained = runs[(source, data)]
                versus_real = (
                    NOT_RUN
                    if data == "real"
                    else difference_with_interval(
                        cross_hter_differences(trained, runs[(source, "real")])
                    )
                )
                versus_one_label = (
                    NOT_RUN
                    if runs is one_label
                    else difference_with_interval(
                        cross_hter_differences(trained, one_label[(source, data)])
                    )
                )
                cells = [
                    title,
                    percent_mean(trained, hter_of("intra")),
                    percent_mean(trained, hter_of("cross")),
                    versus_real,
                    versus_one_label,
                    f"{measured(trained, cross_auc('replay')).mean():.3f}",
                ]
                rows.append("\t\t" + " & ".join(cells) + " \\\\")
        direction = f"from {CORPUS_NAMES[source]} to {CORPUS_NAMES[target]}"
        caption = (
            "Detectors trained with one label per crop, with a depth map instead of the label, "
            "and with a depth map beside the label as an auxiliary task: 7 $\\times$ 7 from "
            "Depth Anything V2 at equal weight, 14 $\\times$ 14, ten times the weight of the "
            "label, and 7 $\\times$ 7 from a 3D face model. All on real, synthetic and mixed "
            "attacks, with the sinusoidal fringes and matched noise: intra-dataset and "
            "cross-dataset HTER (\\%), the paired cross-dataset difference to the real detector "
            "of the same supervision and to the detector with one label trained on the same "
            "attacks, each with its 95\\% bootstrap interval, and the replay AUC. A negative "
            "difference is a lower error than that of the detector it is compared with. Mean over "
            f"ten seeds, {direction}."
            if source == DIRECTIONS[0][0]
            else f"As Table~\\ref{{{first_label}}}, {direction}."
        )
        tables.append(
            table(
                caption,
                f"tab:depth-{source}",
                "l" + CENTRED_X * 2 + "cc" + CENTRED_X,
                "\\textbf{Attacks} & \\textbf{HTER intra} & \\textbf{HTER cross} & "
                "\\textbf{Cross $-$ real} & \\textbf{Cross $-$ one label} & "
                "\\textbf{AUC replay}",
                rows,
            )
        )
    return "\n".join(tables)


def splits_table(splits: Sequence[tuple[str, Runs]]) -> str:
    rows = []
    for title, runs in splits:
        real, mixed = runs[(MSU, "real")], runs[(MSU, "mixed")]
        cells = [
            title,
            development_subjects(real),
            percent_mean(real, hter_of("cross")),
            percent_mean(mixed, hter_of("cross")),
            *paired_difference(mixed, real),
            f"{measured(real, cross_auc('replay')).mean():.3f}",
            f"{measured(mixed, cross_auc('replay')).mean():.3f}",
        ]
        rows.append("\t\t" + " & ".join(cells) + " \\\\")
    rows.append("\t\t\\midrule")
    rows.append("\t\t" + " & ".join(pooled_splits_cells(splits)) + " \\\\")
    return table(
        "From MSU-MFSD to Replay-Mobile, with a different draw of the five training subjects "
        "held out as the development partition: cross-dataset HTER (\\%) of the real and the "
        "mixed detectors with matched noise, the paired difference with its 95\\% bootstrap "
        "interval and the seeds in which the mixed detector improved, and the replay AUC. "
        "A negative difference is a lower error than that of the real detector. "
        "Mean over ten seeds; the last row pools the draws, and its interval resamples the "
        "draws rather than the seeds.",
        "tab:splits",
        "ll" + CENTRED_X * 2 + "cc" + CENTRED_X * 2,
        "& & \\multicolumn{4}{c}{\\textbf{HTER cross}} & "
        "\\multicolumn{2}{c}{\\textbf{AUC replay}} \\\\\n"
        "\t\t\\cmidrule(lr){3-6} \\cmidrule(lr){7-8}\n"
        "\t\t\\textbf{Split} & \\textbf{Development subjects} & \\textbf{Real} & "
        "\\textbf{Mixed} & \\textbf{Mixed $-$ real} & \\textbf{Seeds} & \\textbf{Real} & "
        "\\textbf{Mixed}",
        rows,
    )


def pooled_splits_cells(splits: Sequence[tuple[str, Runs]]) -> list[str]:
    """All draws together: mean errors over every run, and the mean of the per-draw differences
    with an interval that resamples the draws, since the seeds of one draw share its subjects."""
    pairs = [(runs[(MSU, "real")], runs[(MSU, "mixed")]) for _, runs in splits]
    differences = [cross_hter_differences(mixed, real) for real, mixed in pairs]

    def pooled(pick: Callable[[Report], float], index: int) -> np.ndarray:
        return np.concatenate([measured(pair[index], pick) for pair in pairs])

    improved = sum(int(np.sum(draw < 0)) for draw in differences)
    return [
        f"All {len(splits)} draws",
        NOT_RUN,
        f"{100 * pooled(hter_of('cross'), 0).mean():.1f}",
        f"{100 * pooled(hter_of('cross'), 1).mean():.1f}",
        difference_with_interval(np.array([draw.mean() for draw in differences])),
        f"{improved}/{sum(draw.size for draw in differences)}",
        f"{pooled(cross_auc('replay'), 0).mean():.3f}",
        f"{pooled(cross_auc('replay'), 1).mean():.3f}",
    ]


def synthetic_ratio(run: Path) -> float:
    """Synthetic attacks per bona fide training crop; runs from before the ratio flags had one."""
    training = report(run)["training"]
    return float(training.get("synthetic_per_face", 1) * training.get("synthetic_fraction", 1.0))


def runs_by_ratio(folders: Sequence[Path]) -> dict[float, Runs]:
    by_ratio = {}
    for folder in folders:
        runs = find_runs(folder)
        (ratio,) = {
            synthetic_ratio(run)
            for (_, data), seeds in runs.items()
            if data != "real"
            for run in seeds.values()
        }
        by_ratio[ratio] = runs
    return dict(sorted(by_ratio.items()))


def relative_statistics(section: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Ratios of each group to the bona fide crops. Reports written before the calibration
    search, such as the first generator's, nest them under a single key."""
    if "relative_to_bona_fide" in section:
        return dict(section["relative_to_bona_fide"])
    return {
        group: {**section["tone"][group], "chroma_power": section["chroma_power"][group]}
        for group in section["tone"]
    }


def motion_table(reports: Mapping[str, Report]) -> str:
    def row(kind: str) -> str:
        cells = [
            f"{species[kind]['auc']:.3f}" if kind in species else NOT_RUN
            for species in (
                reports[cue][f"{corpus} intra"]["by_species"]
                for cue in MOTION_CUES
                for corpus in CORPUS_NAMES
            )
        ]
        return "\t\t" + " & ".join([kind.capitalize(), *cells]) + " \\\\"

    return table(
        "AUC of the blink and the pulse scores, which need no training, on the test videos of "
        "each dataset, for each kind of attack against the bona fide videos. The synthetic "
        "replays are rendered from every bona fide test video by the generator with matched "
        "noise of its dataset, with one screen for the whole video or a new screen in every "
        "frame. An AUC of 0.5 is chance, and below it the attacks look more alive than the "
        "bona fide videos.",
        "tab:motion",
        "l" + CENTRED_X * 4,
        "& \\multicolumn{2}{c}{\\textbf{Blinks}} & \\multicolumn{2}{c}{\\textbf{Pulse}} \\\\\n"
        "\t\t\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
        "\t\t\\textbf{Attack} & \\textbf{MSU-MFSD} & \\textbf{Replay-Mobile} & "
        "\\textbf{MSU-MFSD} & \\textbf{Replay-Mobile}",
        [*map(row, RECORDED_KINDS), "\t\t\\midrule", *map(row, SYNTHETIC_REPLAYS)],
    )


def calibration_table(first_generator: dict[str, Any]) -> str:
    rows = []
    for corpus, name in CORPUS_NAMES.items():
        statistics = relative_statistics(first_generator[corpus])
        for index, (group, title) in enumerate(
            (("replay", "Real replay"), ("print", "Real print"), (GENERATED, "First generator"))
        ):
            values = [f"{statistics[group][statistic]:.2f}" for statistic in TONE_STATISTICS]
            rows.append("\t\t" + " & ".join([name if index == 0 else "", title, *values]) + " \\\\")
        if corpus != list(CORPUS_NAMES)[-1]:
            rows.append("\t\t\\midrule")
    return table(
        "Image statistics of real and synthetic attacks relative to the bona fide crops of the "
        "same training partition; a value of 1 means no difference.",
        "tab:calibration",
        "Xlccccc",
        "\\textbf{Dataset} & \\textbf{Crops} & \\textbf{Saturation} & \\textbf{Contrast} & "
        "\\textbf{Black level} & \\textbf{White level} & \\textbf{Sharpness}",
        rows,
    )


def style(axis: Any) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
    axis.tick_params(colors=INK, labelsize=8)


def seeds_figure(runs: Runs, path: Path) -> None:
    figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    for axis, (source, target) in zip(axes, DIRECTIONS, strict=True):
        for position, data in enumerate(CONFIGURATIONS):
            values = 100 * measured(runs[(source, data)], hter_of("cross"))
            spread = np.linspace(-0.12, 0.12, values.size)
            axis.scatter(position + spread, values, s=14, color=SERIES_COLOURS[0], zorder=3)
            axis.hlines(values.mean(), position - 0.25, position + 0.25, color=INK, linewidth=2)
        axis.set_xticks(range(len(CONFIGURATIONS)), [d.capitalize() for d in CONFIGURATIONS])
        axis.set_title(f"{CORPUS_NAMES[source]} → {CORPUS_NAMES[target]}", fontsize=9, color=INK)
        style(axis)
    axes[0].set_ylabel("Cross-dataset HTER (%)", fontsize=8, color=INK)
    figure.savefig(path)


def log_odds(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, SCORE_EPSILON, 1.0 - SCORE_EPSILON)
    return np.asarray(np.log(clipped / (1.0 - clipped)))


def representative_seed(runs: dict[int, Path]) -> int:
    """The seed whose cross-dataset HTER is nearest the mean over all seeds."""
    hters = {seed: hter_of("cross")(report(run)) for seed, run in runs.items()}
    mean = float(np.mean(list(hters.values())))
    return min(hters, key=lambda seed: abs(hters[seed] - mean))


def threshold_shift_figure(runs: Runs, path: Path) -> int:
    """Score histograms, in log-odds, of the real detector trained on MSU-MFSD on its own test
    partition and on that of Replay-Mobile, with the one threshold its development partition
    fixed. Returns the seed drawn."""
    source, target = DIRECTIONS[0]
    seed = representative_seed(runs[(source, "real")])
    run = runs[(source, "real")][seed]
    scenarios = (("intra", source), ("cross", target))
    groups = {scenario: partition_scores(run, scenario)[:2] for scenario, _ in scenarios}
    every_score = log_odds(np.concatenate([s for pair in groups.values() for s in pair]))
    edges = np.arange(
        np.floor(every_score.min()),
        np.ceil(every_score.max()) + LOG_ODDS_BIN_WIDTH,
        LOG_ODDS_BIN_WIDTH,
    )
    figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), layout="constrained")
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    for axis, (scenario, corpus) in zip(axes, scenarios, strict=True):
        attacks, bona_fide = groups[scenario]
        for colour, title, scores in (
            (SERIES_COLOURS[0], "Bona fide", bona_fide),
            (SERIES_COLOURS[1], "Attacks", attacks),
        ):
            axis.hist(
                log_odds(scores),
                bins=edges,
                weights=np.full(scores.size, 1.0 / scores.size),
                color=colour,
                alpha=HISTOGRAM_OPACITY,
                label=title,
            )
        threshold = log_odds(np.array(report(run)[scenario]["threshold"]))
        axis.axvline(float(threshold), color=INK, linestyle="--", linewidth=1, label="Threshold")
        axis.set_title(f"{CORPUS_NAMES[corpus]} test partition", fontsize=9, color=INK)
        axis.set_xlabel("Attack score (log-odds)", fontsize=8, color=INK)
        style(axis)
    axes[0].set_ylabel("Share of crops", fontsize=8, color=INK)
    axes[0].legend(fontsize=7, frameon=False)
    figure.savefig(path)
    return seed


def ratio_curves(
    real: Runs, by_ratio: dict[float, Runs]
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    """Mean and standard deviation over the seeds, at each synthetic-per-bona-fide ratio, of
    the cross-dataset replay AUC and HTER (%), by source corpus, measure and training data,
    the real detector repeated at every ratio."""
    measures = (("Replay AUC", cross_auc("replay"), 1.0), ("HTER (%)", hter_of("cross"), 100.0))
    curves = {}
    for source, _ in DIRECTIONS:
        for title, pick, scale in measures:
            for data in ("real", "synthetic", "mixed"):
                values = np.array(
                    [
                        scale * measured((real if data == "real" else runs)[(source, data)], pick)
                        for runs in by_ratio.values()
                    ]
                )
                curves[(source, title, data)] = (values.mean(axis=1), values.std(axis=1))
    return curves


def ratio_figure(
    curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
    ratios: Sequence[float],
    path: Path,
) -> None:
    titles = list(dict.fromkeys(title for _, title, _ in curves))
    figure = Figure(figsize=(FIGURE_WIDTH_IN, 2 * FIGURE_HEIGHT_IN), layout="constrained")
    axes = figure.subplots(len(titles), len(DIRECTIONS), sharex=True, sharey="row")
    for row, title in enumerate(titles):
        for column, (source, target) in enumerate(DIRECTIONS):
            axis = axes[row, column]
            real_mean, _ = curves[(source, title, "real")]
            axis.axhline(float(real_mean[0]), color=INK, linestyle="--", linewidth=1, label="Real")
            for colour, data in zip(SERIES_COLOURS, ("synthetic", "mixed"), strict=False):
                mean, std = curves[(source, title, data)]
                axis.plot(
                    ratios, mean, marker="o", markersize=3, color=colour, label=data.capitalize()
                )
                axis.fill_between(
                    ratios, mean - std, mean + std, color=colour, alpha=BAND_OPACITY, linewidth=0
                )
            axis.set_xscale("log")
            axis.set_xticks(list(ratios), [f"{ratio:g}" for ratio in ratios])
            axis.xaxis.set_minor_formatter(NullFormatter())
            if row == 0:
                axis.set_title(
                    f"{CORPUS_NAMES[source]} → {CORPUS_NAMES[target]}", fontsize=9, color=INK
                )
            if row == len(titles) - 1:
                axis.set_xlabel("Synthetic attacks per bona fide crop", fontsize=8, color=INK)
            style(axis)
        axes[row, 0].set_ylabel(title, fontsize=8, color=INK)
    axes[0, 0].legend(fontsize=7, frameon=False)
    figure.savefig(path)


def print_ratio_curves(
    curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]], ratios: Sequence[float]
) -> None:
    """The values behind fig-ratio, which the text quotes."""
    print("\n== fig-ratio: mean and standard deviation over the seeds")
    print("  ratios " + "  ".join(f"{ratio:g}" for ratio in ratios))
    for (source, title, data), (mean, std) in curves.items():
        cells = "  ".join(f"{m:.3f}±{s:.3f}" for m, s in zip(mean, std, strict=True))
        print(f"  {CORPUS_NAMES[source]:13s} {title:10s} {data:9s} {cells}")


def spectrum_profiles(
    first_generator: dict[str, Any], calibrated: dict[str, dict[str, Any]], corpus: str
) -> list[tuple[str, list[float]]]:
    fitted = relative_statistics(calibrated[corpus][corpus])
    first = relative_statistics(first_generator[corpus])
    return [
        ("Real replay", fitted["replay"]["chroma_power"]),
        ("First generator", first[GENERATED]["chroma_power"]),
        ("Generator calibrated on this dataset", fitted[GENERATED]["chroma_power"]),
    ]


def spectrum_figure(
    first_generator: dict[str, Any], calibrated: dict[str, dict[str, Any]], path: Path
) -> None:
    edges = np.array(first_generator["bands"], dtype=np.float64)
    centres = np.sqrt(edges[:-1] * edges[1:])
    figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    for axis, corpus in zip(axes, CORPUS_NAMES, strict=True):
        for (title, powers), colour, marker, line in zip(
            spectrum_profiles(first_generator, calibrated, corpus),
            SERIES_COLOURS,
            ("o", "s", "^"),
            ("-", "--", ":"),
            strict=True,
        ):
            axis.plot(
                centres,
                powers,
                color=colour,
                marker=marker,
                markersize=4,
                linestyle=line,
                linewidth=1.5,
                label=title,
            )
        axis.axhline(1.0, color=INK, linewidth=0.8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(edges[::2], [f"{edge:.0f}" for edge in edges[::2]])
        axis.set_xlabel("Spatial frequency (cycles per width)", fontsize=8, color=INK)
        axis.set_title(CORPUS_NAMES[corpus], fontsize=9, color=INK)
        style(axis)
    axes[0].set_ylabel("Color power / bona fide", fontsize=8, color=INK)
    axes[0].set_yticks(SPECTRUM_TICKS, [f"{tick:g}" for tick in SPECTRUM_TICKS])
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=7, frameon=False)
    figure.savefig(path)


def bootstrap_interval(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = rng.choice(differences, (BOOTSTRAP_RESAMPLES, differences.size)).mean(axis=1)
    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(resampled, [tail, 1.0 - tail])
    return float(low), float(high)


def on_report(pick: Callable[[Report], float]) -> Callable[[Path], float]:
    return lambda run: pick(report(run))


def print_paired_differences(title: str, runs: Runs) -> None:
    """Each pair of training configurations, compared seed by seed."""
    print(f"\n== {title}")
    measures = (
        ("cross HTER (points)", on_report(hter_of("cross")), 100.0, True),
        (
            "cross HTER, bona fide rule",
            lambda run: bona_fide_rule(run, "cross")["hter"],
            100.0,
            True,
        ),
        ("intra HTER (points)", on_report(hter_of("intra")), 100.0, True),
        ("cross AUC", on_report(cross_auc(None)), 1.0, False),
        ("cross replay AUC", on_report(cross_auc("replay")), 1.0, False),
    )
    comparisons = (("synthetic", "real"), ("mixed", "real"), ("mixed", "synthetic"))
    for source, target in DIRECTIONS:
        for data, baseline in comparisons:
            trained, reference = runs[(source, data)], runs[(source, baseline)]
            seeds = sorted(set(reference) & set(trained))
            print(f"\n{CORPUS_NAMES[source]} -> {CORPUS_NAMES[target]}, {data} against {baseline}")
            for title, pick, scale, lower_is_better in measures:
                differences = scale * np.array(
                    [pick(trained[s]) - pick(reference[s]) for s in seeds]
                )
                low, high = bootstrap_interval(differences)
                improved = int(np.sum(differences < 0 if lower_is_better else differences > 0))
                print(
                    f"  {title:26s} {differences.mean():+8.3f}  "
                    f"interval {low:+.3f} to {high:+.3f}  improved in {improved}/{len(seeds)}"
                )


def print_spectrum_fit(calibrated: dict[str, dict[str, Any]]) -> None:
    for corpus in CORPUS_NAMES:
        statistics = relative_statistics(calibrated[corpus][corpus])
        gap = np.abs(
            np.array(statistics[GENERATED]["chroma_power"])
            / np.array(statistics["replay"]["chroma_power"])
            - 1.0
        )
        name = CORPUS_NAMES[corpus]
        print(f"\n{name}: calibrated generator within {gap.max():.0%} of real replays per band")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render every table and figure of the thesis results from results/, and the "
        "composition of each partition from the face crops in data/frames."
    )
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--clean", type=Path, default=Path("results/clean"))
    parser.add_argument("--noise-off", type=Path, default=Path("results/no-noise"))
    parser.add_argument(
        "--noise-on-real-crops", type=Path, default=Path("results/noise-real-crops")
    )
    parser.add_argument(
        "--synthetic-ratios",
        type=Path,
        nargs="+",
        default=[
            Path("results/fraction-0.1"),
            Path("results/fraction-0.25"),
            Path("results/fraction-0.5"),
            Path("results/noise-real-crops"),
            Path("results/copies-3"),
            Path("results/copies-5"),
        ],
        help="runs with noise on every real crop, one folder per synthetic-per-bona-fide ratio",
    )
    parser.add_argument(
        "--matched-noise",
        type=Path,
        nargs=2,
        metavar=("ONE", "THREE"),
        default=[Path("results/matched-noise-fraction-1"), Path("results/matched-noise-copies-3")],
        help="sinusoidal generator with matched noise, one and three synthetic attacks per crop",
    )
    parser.add_argument(
        "--screen",
        type=Path,
        nargs=2,
        metavar=("ONE", "THREE"),
        default=[Path("results/screen-best-fraction-1"), Path("results/screen-best-copies-3")],
        help="simulated screen with matched noise, one and three synthetic attacks per crop",
    )
    parser.add_argument(
        "--spectral",
        type=Path,
        nargs=2,
        metavar=("MAIN", "NARROW"),
        default=[Path("results/spectral-band-0.07"), Path("results/spectral-band-0.02")],
        help="spectral transfer from the real source replays, with the main and the narrow band",
    )
    parser.add_argument(
        "--msu-splits",
        type=Path,
        nargs="+",
        default=[Path(f"results/msu-split-{draw}") for draw in range(1, MSU_SPLIT_DRAWS + 1)],
        help="real and mixed MSU-MFSD judges with other draws of the development subjects",
    )
    parser.add_argument(
        "--mixstyle",
        type=Path,
        default=Path("results/mixstyle"),
        help="real and mixed ResNet-18 judges trained with MixStyle, matched-noise generator",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=Path("results/depth"),
        help="real, synthetic and mixed ResNet-18 judges supervised with pseudo-depth maps, "
        "matched-noise generator",
    )
    parser.add_argument(
        "--depth-auxiliary",
        type=Path,
        default=Path("results/depth-auxiliary"),
        help="the same judges with the depth map as an auxiliary task beside the attack label",
    )
    parser.add_argument(
        "--depth-variants",
        type=Path,
        nargs=3,
        metavar=("GRID_14", "WEIGHT_10", "FACE_MODEL"),
        default=[
            Path("results/depth-14"),
            Path("results/depth-weight10"),
            Path("results/depth-face-model"),
        ],
        help="the auxiliary judges with a 14 x 14 map from layer3, with the depth loss at ten "
        "times the weight of the label, and with targets from a 3D face model",
    )
    parser.add_argument(
        "--vit",
        type=Path,
        nargs=2,
        metavar=("B16", "B32"),
        default=[Path("results/vit-matched-noise"), Path("results/vit-b32-matched-noise")],
        help="ViT-B/16 and ViT-B/32 judges, real, synthetic and mixed, matched-noise generator",
    )
    parser.add_argument("--first-generator-runs", type=Path, default=Path("results/judges"))
    parser.add_argument(
        "--first-generator-calibration",
        type=Path,
        default=Path("results/calibration-first-generator.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        nargs=2,
        metavar=("MSU", "REPLAY_MOBILE"),
        default=[Path("results/calibration.json"), Path("results/calibration-fitted-rm.json")],
        help="calibration reports of the generator fitted to each dataset",
    )
    parser.add_argument(
        "--motion-cues",
        type=Path,
        nargs=2,
        metavar=("BLINKS", "PULSE"),
        default=[Path("results/blinks"), Path("results/pulse")],
        help="folders of the reports of measure_blinks.py and measure_pulse.py",
    )
    parser.add_argument("--output", type=Path, default=Path("monographs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    clean = find_runs(args.clean)
    real_only = {key: runs for key, runs in clean.items() if key[1] == "real"}
    noise_off = {**find_runs(args.noise_off), **real_only}
    noise_on_real_crops = find_runs(args.noise_on_real_crops)
    by_ratio = runs_by_ratio(args.synthetic_ratios)
    matched_one, matched_three = (find_runs(path) for path in args.matched_noise)
    screen_one, screen_three = (find_runs(path) for path in args.screen)
    spectral_main, spectral_narrow = (find_runs(path) for path in args.spectral)
    vit_b16, vit_b32 = (find_runs(path) for path in args.vit)
    depth = find_runs(args.depth)
    depth_auxiliary = find_runs(args.depth_auxiliary)
    depth_variants = [
        ("14 $\\times$ 14 map", "a 14 x 14 map", find_runs(args.depth_variants[0])),
        ("weight 10", "the depth loss at weight 10", find_runs(args.depth_variants[1])),
        ("3D face model", "targets from a 3D face model", find_runs(args.depth_variants[2])),
    ]
    splits = [
        ("Original", {**real_only, (MSU, "mixed"): matched_one[(MSU, "mixed")]}),
        *((f"Split {index}", find_runs(path)) for index, path in enumerate(args.msu_splits, 1)),
    ]
    first_generator_runs = find_runs(args.first_generator_runs)
    first_generator = json.loads(args.first_generator_calibration.read_text())
    calibrated = {
        corpus: json.loads(path.read_text())
        for corpus, path in zip(CORPUS_NAMES, args.calibration, strict=True)
    }
    motion_reports = {
        cue: json.loads((folder / "report.json").read_text())
        for cue, folder in zip(MOTION_CUES, args.motion_cues, strict=True)
    }
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tab-composition.tex": composition_table(load_partitions(args.frames)),
        "tab-baseline.tex": baseline_table(clean),
        "tab-main.tex": main_table(clean),
        "tab-metrics.tex": metrics_table(clean),
        "tab-threshold.tex": threshold_table(clean),
        "tab-noise.tex": noise_table(clean, find_runs(args.noise_off), noise_on_real_crops),
        "tab-first-generator.tex": first_generator_table(first_generator_runs),
        "tab-ablation.tex": ablation_table(first_generator_runs),
        "tab-ablation-noise.tex": noise_ablation_table(clean, noise_on_real_crops),
        "tab-generators.tex": generators_table(
            clean,
            (
                ("Sinusoidal, fixed noise", clean),
                ("Sinusoidal, matched noise", matched_one),
                ("\\quad three per crop", matched_three),
                ("Screen, matched noise", screen_one),
                ("\\quad three per crop", screen_three),
                ("Spectral transfer, 0.07 cycles/px", spectral_main),
                ("\\quad 0.02 cycles/px", spectral_narrow),
            ),
        ),
        "tab-ablation-generators.tex": generators_ablation_table(matched_one, screen_one),
        "tab-architectures.tex": architectures_table(
            (
                ("ResNet-18", {**matched_one, **real_only}),
                ("ViT-B/16", vit_b16),
                ("ViT-B/32", vit_b32),
            )
        ),
        "tab-mixstyle.tex": mixstyle_table(clean, matched_one, find_runs(args.mixstyle)),
        "tab-depth.tex": depth_table(
            {**matched_one, **real_only},
            (
                ("\\quad depth instead of label", depth),
                ("\\quad depth beside label", depth_auxiliary),
                *((f"\\quad\\quad {label}", runs) for label, _, runs in depth_variants),
            ),
        ),
        "tab-splits.tex": splits_table(splits),
        "tab-calibration.tex": calibration_table(first_generator),
        "tab-motion.tex": motion_table(motion_reports),
    }
    for name, content in outputs.items():
        (args.output / name).write_text(content)
    seeds_figure(clean, args.output / "fig-seeds-hter.pdf")
    spectrum_figure(first_generator, calibrated, args.output / "fig-colour-spectrum.pdf")
    shift_seed = threshold_shift_figure(clean, args.output / "fig-threshold-shift.pdf")
    curves = ratio_curves(noise_on_real_crops, by_ratio)
    ratio_figure(curves, list(by_ratio), args.output / "fig-ratio.pdf")
    print(
        f"wrote {', '.join(outputs)}, fig-seeds-hter.pdf, fig-colour-spectrum.pdf, "
        f"fig-threshold-shift.pdf (seed {shift_seed}) and fig-ratio.pdf"
    )
    print_ratio_curves(curves, list(by_ratio))
    print_paired_differences("noise stage on, as in tab-main", clean)
    print_paired_differences("noise stage off", noise_off)
    print_paired_differences("noise on every real training crop", noise_on_real_crops)
    real_on_real_crops = {
        key: runs for key, runs in noise_on_real_crops.items() if key[1] == "real"
    }
    for ratio, runs in by_ratio.items():
        if ratio != 1.0:
            print_paired_differences(
                f"{ratio:g} synthetic per bona fide", {**runs, **real_on_real_crops}
            )
    for title, runs in (
        ("sinusoidal, matched noise", matched_one),
        ("sinusoidal, matched noise, three per crop", matched_three),
        ("screen, matched noise", screen_one),
        ("screen, matched noise, three per crop", screen_three),
        ("spectral transfer, 0.07 cycles per pixel", spectral_main),
        ("spectral transfer, 0.02 cycles per pixel", spectral_narrow),
    ):
        print_paired_differences(title, {**runs, **real_only})
    print_paired_differences("ViT-B/16, sinusoidal, matched noise", vit_b16)
    print_paired_differences("ViT-B/32, sinusoidal, matched noise", vit_b32)
    print_paired_differences("ResNet-18 with depth maps, sinusoidal, matched noise", depth)
    print_paired_differences(
        "ResNet-18 with a depth map beside the label, sinusoidal, matched noise", depth_auxiliary
    )
    for _, description, runs in depth_variants:
        print_variant_differences(
            f"Depth beside the label with {description}, against the 7 x 7 map at weight 1",
            runs,
            depth_auxiliary,
        )
    print_spectrum_fit(calibrated)


if __name__ == "__main__":
    main()
