"""Aggregate comparable held-out test results across independent training seeds."""

import argparse
import json
from pathlib import Path

import numpy as np

from dust.data import write_json
from dust.resume import validation_schedule


def summarize_runs(paths):
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if len(reports) < 3:
        raise ValueError("Use at least three completed training seeds")
    if any(r["split"] != "test" or r["is_smoke_run"] for r in reports):
        raise ValueError("Only full held-out test evaluations qualify")
    seeds = [r["training_seed"] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be distinct")
    for key in ("manifest_fingerprint", "threshold", "frames", "checkpoint_source",
                "evaluation_seed", "evaluation_batch_size", "training_budget_epochs"):
        if any(r[key] != reports[0][key] for r in reports):
            raise ValueError(f"Runs use incompatible {key}")
    configurations = [{k: v for k, v in r["training_config"].items() if k != "seed"}
                      for r in reports]
    if any(c != configurations[0] for c in configurations):
        raise ValueError("Training configurations differ beyond the random seed")
    schedules = [r.get("validation_schedule", validation_schedule(r["training_config"],
                                                                 r.get("resume_lineage", []))) for r in reports]
    if any(schedule != schedules[0] for schedule in schedules):
        raise ValueError("Runs use incompatible validation schedule histories")
    result = {"seeds": seeds, "n_runs": len(reports),
              "manifest_fingerprint": reports[0]["manifest_fingerprint"],
              "threshold": reports[0]["threshold"], "metrics": {}}
    for key in ("f1", "precision", "recall", "dust_iou", "false_deletion_rate"):
        values = [r["overall"][key] for r in reports]
        if any(v is None for v in values):
            raise ValueError(f"Undefined metric in one run: {key}")
        result["metrics"][key] = {"mean": float(np.mean(values)),
                                  "sample_std": float(np.std(values, ddof=1)), "per_seed": values}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize_runs(args.metrics)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
