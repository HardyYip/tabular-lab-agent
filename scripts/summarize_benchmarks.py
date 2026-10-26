"""Curate local reference runs and produce a publication-ready comparison figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

from tabular_lab.benchmark import curate_run
from tabular_lab.io import read_json
from tabular_lab.runner import _report

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
        source = Path(row["run_path"])
        destination = args.output / "runs" / label
        curate_run(source, destination)
        _report(destination, read_json(source / "summary.json"), read_json(source / "experiments.json"))
    public = [{k: v for k, v in row.items() if k != "run_path"} for row in rows]
    frame = pd.DataFrame(public)
    frame.to_csv(args.output / "results.csv", index=False)
    (args.output / "results.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    completed = frame.loc[frame.status == "completed"]
    aggregate = completed.groupby(["task", "strategy"]).agg(
        n=("seed", "count"), macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"), seconds_mean=("seconds", "mean"),
        cost_mean=("cost_cny_estimate", "mean"), trials=("trials", "sum"),
        failed_trials=("failed_trials", "sum"))
    aggregate["trial_failure_rate"] = aggregate.failed_trials / aggregate.trials
    aggregate.to_csv(args.output / "aggregate.csv")
    tasks = list(completed.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), squeeze=False)
    labels = {"baseline": "Fixed baseline\n(2 candidates)", "random": "Random search\n(up to 8)",
              "agent": "LLM Agent\n(up to 8)", "offline_mock_agent": "Offline mock\n(up to 8)",
              "codegen": "Code generation"}
    colors = {"baseline": "#6c859d", "random": "#2a9d8f", "agent": "#de8f42"}
    for ax, task in zip(axes[0], tasks, strict=True):
        available = aggregate.loc[task]
        values = available.loc[[s for s in labels if s in available.index]]
        x = range(len(values))
        ax.bar(x, values.macro_f1_mean, yerr=values.macro_f1_std.fillna(0),
               color=[colors.get(s, "#8876b9") for s in values.index], capsize=5, width=0.55)
        ax.set_xticks(list(x), [labels.get(s, s) for s in values.index])
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_title(task, pad=12)
        ax.set_ylabel("Final-test macro-F1")
        ax.spines[["top", "right"]].set_visible(False)
        for index, score in enumerate(values.macro_f1_mean):
            ax.text(index, score + 0.035, f"{score:.3f}", ha="center", fontsize=10)
    calls = int(frame.llm_calls.sum())
    total_cost = frame.cost_cny_estimate.sum()
    seeds = sorted(frame.seed.unique().tolist())
    fig.suptitle(f"Classification experiments · {len(seeds)} seeds · {calls} LLM calls", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    lines = ["# Classification benchmark", "", f"Seeds: {seeds}; {calls} LLM calls; estimated total "
             f"cost CNY {total_cost:.6f}. Fixed baselines use two candidates; random search and the "
             "V1 Agent share the same finite candidate space and up to eight candidates per run. "
             "All use the same split/seed protocol and resource ceilings. Mock/replay records are not live calls.", "",
             "![Comparison](comparison.png)", "",
             "| Dataset | Strategy | Seeds | Test macro-F1 mean ± std | Mean seconds | Mean CNY | Trial failure rate |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (task, strategy), row in aggregate.iterrows():
        lines.append(f"| {task} | {strategy} | {int(row.n)} | {row.macro_f1_mean:.4f} ± "
                     f"{row.macro_f1_std:.4f} | {row.seconds_mean:.1f} | {row.cost_mean:.6f} | "
                     f"{row.trial_failure_rate:.1%} |")
    lines.extend(["", "Wine Quality is our predefined three-class task after duplicate handling. "
                  "Adult preserves the official test with a cleaned development set. "
                  "GPU trials may show small numerical differences across reruns. "
                  "See ../../docs/DATASETS.md for exact protocols.", "",
                  "Manifests record each run's actual commit, dirty state, source hash and package versions. "
                  "Early development runs may honestly have a null commit. Failed/partial runs remain in "
                  "results.csv but are excluded from the score means above. Test metrics are used only "
                  "after validation-based selection; repeated development on these tasks limits generalization claims. "
                  "Public reports are rendered from the recorded summaries; raw logs remain private. "
                  "Tests and validation jobs ran concurrently, so recorded runtimes are indicative rather than "
                  "controlled speed measurements. Three seeds support a descriptive comparison, not a significance claim."])
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(aggregate.to_string())


if __name__ == "__main__":
    main()
