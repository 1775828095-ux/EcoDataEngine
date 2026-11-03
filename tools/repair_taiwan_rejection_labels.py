"""Repair only empty-window rejection metadata; preserve original runs and all predictions."""
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ["taiwan_contract_sensitivity_20261019", "taiwan_contract_sensitivity_paired_20261019"]:
        folder = ROOT / "output" / name
        target = folder / "executions_window_labels_corrected.csv"
        if target.exists():
            raise ValueError("Preserve existing correction evidence")
        runs, coverage = pd.read_csv(folder / "executions.csv"), pd.read_csv(folder / "coverage.csv")
        corrections = []
        for hypothesis, windows in coverage.groupby("hypothesis", sort=False):
            indices = list(runs.index[runs.hypothesis.eq(hypothesis)])
            offset = 0
            for window in windows.itertuples():
                count = 6 if window.in_hull_qualified else 1
                selected = indices[offset:offset+count]
                assert len(selected) == count
                for index in selected:
                    old = runs.loc[index, "time"]
                    if old != window.time:
                        assert runs.loc[index, "status"] == "INSUFFICIENT_INPUT"
                        assert pd.isna(runs.loc[index, "method"])
                        corrections.append(dict(row=int(index), hypothesis=hypothesis, old_time=old, corrected_time=window.time))
                        runs.loc[index, "time"] = window.time
                        runs.loc[index, "reused_from"] = "EMPTY_REJECTION_METADATA_CORRECTED_NOT_NUMERICAL_EXECUTION"
                offset += count
            assert offset == len(indices)
        runs.to_csv(target, index=False)
        (folder / "empty_window_label_repair.json").write_text(json.dumps(dict(
            cause="Empty table cache fingerprint omitted requested stamp", corrections=corrections,
            predictions_changed=False, algorithms_rerun=False, original_execution_log_preserved=True), indent=2), encoding="utf-8")
        print(name, "rejection labels corrected", len(corrections), flush=True)


if __name__ == "__main__":
    main()
