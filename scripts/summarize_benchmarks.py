"""Curate local reference runs and produce a publication-ready comparison figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

from tabular_lab.benchmark import curate_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("runs/benchmark-local"))
    parser.add_argument("--output", type=Path, default=Path("examples/benchmark-local"))
    args = parser.parse_args()
    rows = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        label = f"{row['task']}-{row['strategy']}-seed{row['seed']}"
        curate_run(Path(row["run_path"]), args.output / "runs" / label)
    public = [{k: v for k, v in row.items() if k != "run_path"} for row in rows]
    frame = pd.DataFrame(public)
    frame.to_csv(args.output / "results.csv", index=False)
    (args.output / "results.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    completed = frame.loc[frame.status == "completed"]
    aggregate = completed.groupby(["task", "strategy"]).agg(
        n=("seed", "count"), macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"), seconds_mean=("seconds", "mean"),
        failed_trials=("failed_trials", "sum"))
    aggregate.to_csv(args.output / "aggregate.csv")
    tasks = list(completed.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(8.4, 4), squeeze=False)
    for ax, task in zip(axes[0], tasks, strict=True):
        values = aggregate.loc[task]
        x = range(len(values))
        ax.bar(x, values.macro_f1_mean, yerr=values.macro_f1_std.fillna(0),
               color=["#6c859d", "#2a9d8f"][:len(values)], capsize=5, width=0.55)
        ax.set_xticks(list(x), ["Fixed baseline\n(2 candidates)" if s == "baseline" else
                              "Random search\n(8 candidates)" for s in values.index])
        ax.set_ylim(0, 1)
        ax.set_title(task)
        ax.set_ylabel("Final-test macro-F1")
        ax.spines[["top", "right"]].set_visible(False)
        for index, score in enumerate(values.macro_f1_mean):
            ax.text(index, score + 0.035, f"{score:.3f}", ha="center", fontsize=10)
    fig.suptitle("Local reference experiments · 3 seeds · no LLM calls", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    lines = ["# Local reference benchmark", "", "Three seeds (17, 42, 73); no LLM calls or API cost. "
             "Fixed baselines use two candidates; random search uses up to eight. "
             "These are reference experiments, not evidence of LLM effectiveness.", "",
             "![Comparison](comparison.png)", "", "| Dataset | Strategy | Seeds | Test macro-F1 mean ± std | Mean seconds |",
             "| --- | --- | ---: | ---: | ---: |"]
    for (task, strategy), row in aggregate.iterrows():
        lines.append(f"| {task} | {strategy} | {int(row.n)} | {row.macro_f1_mean:.4f} ± "
                     f"{row.macro_f1_std:.4f} | {row.seconds_mean:.1f} |")
    lines.extend(["", "Wine Quality is our predefined three-class task after duplicate handling. "
                  "Adult preserves the official test with a cleaned development set. "
                  "GPU trials may show small numerical differences across reruns. "
                  "See ../../docs/DATASETS.md for exact protocols.", "",
                  "These first development runs began before the initial Git commit, so their manifests "
                  "honestly contain a null commit. The V1 experiment/data/search code is included with this "
                  "release. Future runs record the current commit and package-source hash."])
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(aggregate.to_string())


if __name__ == "__main__":
    main()
