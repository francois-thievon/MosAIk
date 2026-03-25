#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

from extreme_ratio_sweep_v1 import (
    ExperimentConfig,
    aggregate_results,
    plot_results,
    summarize_repeat_linearity,
)


ResultValue = Union[float, int]
ResultRow = Dict[str, ResultValue]


def _parse_row(raw: Dict[str, str]) -> ResultRow:
    return {
        "repeat": int(raw["repeat"]),
        "dim": int(raw["dim"]),
        "bottleneck": int(raw["bottleneck"]),
        "ratio": float(raw["ratio"]),
        "spearman": float(raw["spearman"]),
        "pvalue": float(raw["pvalue"]),
        "final_loss": float(raw["final_loss"]),
    }


def load_raw_results(csv_path: Path) -> List[ResultRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_parse_row(row) for row in reader]


def save_csv(path: Path, rows: List[ResultRow]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detect_completed_repeats(raw_results: List[ResultRow]) -> Tuple[int, int, int]:
    repeat_counts = Counter(int(row["repeat"]) for row in raw_results)
    if not repeat_counts:
        return 0, 0, 0

    # In this sweep, a repeat is expected to have one row per tested ratio.
    ratios_per_repeat = max(repeat_counts.values())
    completed_repeats = sum(1 for count in repeat_counts.values() if count == ratios_per_repeat)
    partial_repeats = sum(1 for count in repeat_counts.values() if count < ratios_per_repeat)
    return completed_repeats, partial_repeats, ratios_per_repeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruit les sorties (plots + resumes) de extreme_ratio_sweep_v1 "
            "a partir d'un raw_results_partial.csv."
        )
    )
    default_input = Path(__file__).resolve().parent / "extreme_ratio_sweep_v1_outputs" / "run_159664_1" / "raw_results_partial.csv"
    parser.add_argument("--input-csv", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV introuvable: {input_csv}")

    output_dir = args.output_dir.resolve() if args.output_dir else input_csv.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results = load_raw_results(input_csv)
    if not raw_results:
        raise ValueError("Le CSV fourni est vide.")

    completed_repeats, partial_repeats, ratios_per_repeat = detect_completed_repeats(raw_results)

    aggregated = aggregate_results(raw_results)
    linearity_summary = summarize_repeat_linearity(raw_results)

    save_csv(output_dir / "raw_results_replayed.csv", raw_results)
    save_csv(output_dir / "aggregated_results_replayed.csv", aggregated)

    config = ExperimentConfig(
        output_dir=output_dir,
        n_samples=args.samples,
        repeats=completed_repeats if completed_repeats > 0 else len({int(row["repeat"]) for row in raw_results}),
        show_plot=not args.no_show,
    )

    plot_results(
        raw_results=raw_results,
        aggregated=aggregated,
        linearity_summary=linearity_summary,
        output_path=output_dir / "extreme_ratio_sweep_replayed.png",
        config=config,
    )

    print("Resume linearite semi-log (replay):")
    print(
        f"R2 moyen = {linearity_summary['r2_mean']:.4f} +- {linearity_summary['r2_std']:.4f} | "
        f"Q10 = {linearity_summary['r2_q10']:.4f} | Mediane = {linearity_summary['r2_q50']:.4f} | "
        f"Q90 = {linearity_summary['r2_q90']:.4f}"
    )
    print(
        f"Pente moyenne = {linearity_summary['slope_mean']:.4f} +- {linearity_summary['slope_std']:.4f}"
    )
    print(
        f"Repeats detectes: {completed_repeats} complets"
        + (f", {partial_repeats} partiel(s)" if partial_repeats else "")
        + f" | Ratios par repeat: {ratios_per_repeat}"
    )
    print(f"Sorties ecrites dans: {output_dir}")


if __name__ == "__main__":
    main()
