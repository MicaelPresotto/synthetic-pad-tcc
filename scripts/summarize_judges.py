import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLUMNS = ("hter", "auc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mean and standard deviation over seeds of every judge report in a folder."
    )
    parser.add_argument("--results", type=Path, default=Path("results/judges"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    seeds: dict[tuple[str, str], int] = defaultdict(int)
    for path in sorted(args.results.glob("*/report.json")):
        report = json.loads(path.read_text())
        training = report["training"]
        run = (training["corpus"], training["data"])
        seeds[run] += 1
        for scenario in ("intra", "cross"):
            section = report[scenario]
            for attacks, metrics in (
                ("all", section["all_attacks"]),
                *section["by_species"].items(),
            ):
                for column in COLUMNS:
                    key = (*run, f"{scenario} -> {section['corpus']}", f"{attacks} {column}")
                    grouped[key].append(metrics[column])

    for (corpus, data), count in sorted(seeds.items()):
        print(f"\ntrained on {corpus} ({data}), {count} seeds")
        for (run_corpus, run_data, scenario, metric), values in sorted(grouped.items()):
            if (run_corpus, run_data) == (corpus, data):
                print(f"  {scenario:24s} {metric:12s} {np.mean(values):.3f} ± {np.std(values):.3f}")


if __name__ == "__main__":
    main()
