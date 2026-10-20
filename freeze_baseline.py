"""Freeze the repaired baseline protocol and emit resumable server commands."""

import argparse
import json
from pathlib import Path
import shlex

import torch

from dust.data import file_sha256, fingerprint, read_manifest, write_json
from dust.grouping import CAPTURE_POLICY
from dust.splitting import SPLIT_POLICY, split_quality
from train_dust import source_version


SETTINGS = {"epochs": 100, "batch_size": 32, "workers": 4, "samples_per_frame": 4,
            "lr": .001, "weight_decay": .0001, "threshold": .5,
            "eval_every": 10, "eval_batch_size": 4, "eval_backend": "cuda-graph",
            "save_every_steps": 50, "log_every": 25, "cache_frames": 16, "prefetch_factor": 2}
GEOMETRY = {"npoint": 4096, "block_size": 10., "stride": 5., "coord_scale": 10.}


def check_baseline_manifest(manifest):
    if (manifest.get("preparation_protocol") != CAPTURE_POLICY or manifest["geometry"] != GEOMETRY
            or manifest.get("split_protocol") != SPLIT_POLICY or manifest.get("split_seed") != 42
            or any(abs(a - b) > 1e-8 for a, b in zip(manifest.get("requested_split_ratios", []), (.7, .15, .15)))
            or len(manifest.get("requested_split_ratios", [])) != 3):
        raise ValueError("Freeze requires the coverage-balanced v3 manifest, seed 42 and baseline geometry/ratios")
    split_quality(manifest["frames"])


def check_protocol(path):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    value = dict(protocol)
    digest = value.pop("fingerprint")
    if fingerprint(value) != digest:
        raise ValueError("Frozen protocol fingerprint mismatch")
    if (protocol["version"] != 1 or protocol["name"] != "clean_msg_v3"
            or protocol.get("split_policy") != SPLIT_POLICY
            or protocol["settings"] != SETTINGS or protocol["seeds"] != [42, 43, 44]
            or protocol["source"] != source_version() or protocol["source"]["tracked_files_dirty"] is not False):
        raise ValueError("Use the exact clean frozen source/settings/seeds")
    manifest = read_manifest(protocol["manifest_path"])
    check_baseline_manifest(manifest)
    if manifest["fingerprint"] != protocol["manifest_fingerprint"]:
        raise ValueError("Frozen manifest or capture policy changed")
    return protocol, manifest


def check_training_config(config, protocol, seed):
    expected = {key: SETTINGS[key] for key in ("batch_size", "samples_per_frame", "lr", "weight_decay",
                                              "threshold", "eval_every", "eval_batch_size")}
    expected.update(seed=seed, manifest_fingerprint=protocol["manifest_fingerprint"],
                    source=protocol["source"], model="pointnet2_dust_msg", augmentation="z_rotation",
                    max_train_frames=None, max_val_frames=None, evaluation_seed=42,
                    is_smoke_run=False, training_order="epoch_permutation_v1", initialization=None)
    if seed not in protocol["seeds"] or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Run differs from frozen baseline configuration")


def checkpoint_status(path, protocol, seed):
    if not Path(path).exists():
        return "missing"
    state = torch.load(path, map_location="cpu", weights_only=False)
    check_training_config(state["config"], protocol, seed)
    if (state["format_version"] != 2 or state["training_budget_epochs"] != SETTINGS["epochs"]
            or state.get("resume_lineage", []) or state.get("eval_backend") != SETTINGS["eval_backend"]):
        raise ValueError("Run has a different budget, backend or migration history")
    if state["next_epoch"] == SETTINGS["epochs"] and state["next_batch"] == 0:
        return "complete"
    if not 0 <= state["next_epoch"] < SETTINGS["epochs"]:
        raise ValueError("Invalid frozen run cursor")
    return "partial"


def check_report(path, weights, protocol, manifest, seed):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    check_training_config(report["training_config"], protocol, seed)
    if (report["split"] != "test" or report["is_smoke_run"] or report["training_seed"] != seed
            or report["training_budget_epochs"] != SETTINGS["epochs"]
            or report["checkpoint_source"] != protocol["source"]
            or report["threshold"] != SETTINGS["threshold"]
            or report["evaluation_batch_size"] != SETTINGS["eval_batch_size"]
            or report["evaluation_seed"] != 42
            or report["manifest_fingerprint"] != protocol["manifest_fingerprint"]
            or report.get("checkpoint_sha256") != file_sha256(weights)
            or report["frames"] != sum(r["split"] == "test" for r in manifest["frames"])
            or report["inference_backend"] != SETTINGS["eval_backend"]
            or report["validation_schedule"] != {"initial_eval_every": 10, "changes": []}):
        raise ValueError("Existing test report does not match this frozen run")


def server_script(protocol_path, data_root):
    root = Path(__file__).resolve().parent
    quote = shlex.quote
    separator = " " + chr(92) + "\n      "
    flags = separator.join(f"--{key.replace('_', '-')} {value}" for key, value in SETTINGS.items())
    return f'''#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; rerun this script to resume committed batches."; exit 130' INT
cd {quote(str(root))}
PROTO={quote(str(Path(protocol_path).resolve()))}
DATA_ROOT={quote(str(Path(data_root).resolve()))}
MANIFEST={quote(str(json.loads(Path(protocol_path).read_text(encoding="utf-8"))["manifest_path"]))}
PYTHON="${{PYTHON:-python}}"
"$PYTHON" freeze_baseline.py --check "$PROTO"
mkdir -p logs
for seed in 42 43 44; do
  run="runs/clean_msg_v3_seed${{seed}}"
  out="outputs/clean_msg_v3_seed${{seed}}_test"
  status=$("$PYTHON" freeze_baseline.py --check "$PROTO" --status-checkpoint "$run/last.pt" --seed "$seed")
  if [ "$status" != complete ]; then
    resume_args=()
    if [ "$status" = partial ]; then resume_args=(--resume "$run/last.pt"); fi
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u train_dust.py \\
      --data-root "$DATA_ROOT" --manifest "$MANIFEST" --output "$run" --seed "$seed" \\
      {flags} \\
      "${{resume_args[@]}}" 2>&1 | tee -i -a "logs/clean_msg_v3_seed${{seed}}.log"
    status=$("$PYTHON" freeze_baseline.py --check "$PROTO" --status-checkpoint "$run/last.pt" --seed "$seed")
    if [ "$status" != complete ]; then echo "Training paused; rerun to resume."; exit 0; fi
  fi
  if [ -f "$out/metrics.json" ]; then
    "$PYTHON" freeze_baseline.py --check "$PROTO" --check-report "$out/metrics.json" --weights "$run/best.pt" --seed "$seed"
  elif [ -e "$out" ]; then
    echo "Incomplete evaluation output: $out. Preserve/rename it before retrying." >&2
    exit 1
  else
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u evaluate_dust.py \\
      --data-root "$DATA_ROOT" --manifest "$MANIFEST" --checkpoint "$run/best.pt" \\
      --split test --backend cuda-graph --visualize-per-sensor 1 --output "$out" \\
      2>&1 | tee -i -a "logs/clean_msg_v3_seed${{seed}}_test.log"
  fi
done
"$PYTHON" summarize_dust.py \\
  outputs/clean_msg_v3_seed42_test/metrics.json \\
  outputs/clean_msg_v3_seed43_test/metrics.json \\
  outputs/clean_msg_v3_seed44_test/metrics.json \\
  --output outputs/clean_msg_v3_three_seed_summary.json
'''


def freeze(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Protocol output already exists; use a new directory")
    manifest = read_manifest(args.manifest)
    diagnostic = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
    source = source_version()
    check_baseline_manifest(manifest)
    if source["tracked_files_dirty"] is not False or not source["commit"]:
        raise ValueError("Freeze requires the clean repaired manifest, baseline geometry and committed source")
    if (not diagnostic.get("is_diagnostic") or not diagnostic.get("all_review_predictions_available")
            or not diagnostic.get("all_review_confusions_match")
            or not any(case.get("selected_points", 0) > 0 for case in diagnostic.get("cases", []))
            or diagnostic.get("input_fingerprint") != manifest.get("input_fingerprint")):
        raise ValueError("Run matching legacy failure diagnostics with per-point review before freezing")
    protocol = {"version": 1, "name": "clean_msg_v3", "source": source,
                "manifest_path": str(Path(args.manifest).resolve()), "manifest_fingerprint": manifest["fingerprint"],
                "input_fingerprint": manifest["input_fingerprint"], "capture_policy": CAPTURE_POLICY,
                "split_policy": SPLIT_POLICY,
                "diagnostic_sha256": file_sha256(args.diagnostics), "seeds": [42, 43, 44], "settings": SETTINGS,
                "selection": "maximum full validation F1 at fixed 0.5; every 10 epochs and final epoch",
                "scope": "repaired acquisition-day-disjoint binary baseline; no physical-site generalization claim",
                "structure_phase": "separate branch after diagnostic interpretation and baseline; "
                                   "same frozen manifest/seeds/budget, new validation only for model selection"}
    protocol["fingerprint"] = fingerprint(protocol)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", protocol)
    (output / "run_baseline.sh").write_text(server_script(output / "protocol.json", args.data_root), encoding="utf-8")
    print(json.dumps({"protocol": str(output / "protocol.json"), "fingerprint": protocol["fingerprint"],
                      "run": f"bash {output / 'run_baseline.sh'}"}), flush=True)
    return protocol


def main(args):
    if args.check:
        protocol, manifest = check_protocol(args.check)
        if args.status_checkpoint:
            print(checkpoint_status(args.status_checkpoint, protocol, args.seed))
        elif args.check_report:
            if not args.weights:
                raise ValueError("--weights required for report verification")
            check_report(args.check_report, args.weights, protocol, manifest, args.seed)
        else:
            print("Frozen protocol/source/manifest verified", flush=True)
        return
    if not all((args.data_root, args.manifest, args.diagnostics, args.output)):
        raise ValueError("Freeze needs --data-root --manifest --diagnostics --output")
    return freeze(args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "manifest", "diagnostics", "output", "check", "status-checkpoint",
                 "check-report", "weights"):
        parser.add_argument("--" + name)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
