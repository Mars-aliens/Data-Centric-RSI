"""Print a compact report for an experiment 1A result.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


T95_BY_DF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    if data.get("status") != "completed":
        raise SystemExit(f"cannot analyze status={data.get('status')!r}; require a completed run")
    if data.get("data", {}).get("final_test_accessed") is not False:
        raise SystemExit("invalid result: final_test_accessed must be false")
    print("status:", data.get("status"))
    print("protocol:", data.get("protocol"))
    print("final_test_accessed:", data.get("data", {}).get("final_test_accessed"))
    print("strategy                    pairwise_accuracy  mean_target_nll  train_seconds")
    by_strategy = {}
    for row in data.get("results", []):
        metrics = row.get("training", {})
        accuracy = metrics.get("pairwise_accuracy")
        if accuracy is not None:
            by_strategy.setdefault(row.get("strategy"), {})[row.get("seed")] = accuracy
        accuracy_text = "NA" if accuracy is None else f"{accuracy:.4f}"
        print(
            f"{row.get('strategy', ''):<27} {accuracy_text:>17} "
            f"{metrics.get('mean_target_nll', float('nan')):>16.4f} "
            f"{metrics.get('training_seconds', float('nan')):>13.1f}"
        )
    baseline = by_strategy.get("seed-only", {})
    print("\nPaired development pairwise-accuracy deltas vs seed-only (descriptive run-seed t intervals)")
    for strategy, values in by_strategy.items():
        if strategy == "seed-only":
            continue
        paired = [values[seed] - baseline[seed] for seed in sorted(values.keys() & baseline.keys())]
        if not paired:
            print(f"{strategy}: no paired seeds")
            continue
        mean = sum(paired) / len(paired)
        if len(paired) > 1:
            variance = sum((value - mean) ** 2 for value in paired) / (len(paired) - 1)
            margin = T95_BY_DF.get(len(paired) - 1, 1.96) * math.sqrt(variance / len(paired))
            interval = f"[{mean - margin:+.4f}, {mean + margin:+.4f}]"
        else:
            interval = "not estimable (n=1)"
        print(f"{strategy}: delta={mean:+.4f} 95% CI={interval} n={len(paired)} deltas={paired}")


if __name__ == "__main__":
    main()
