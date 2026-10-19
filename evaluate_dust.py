"""Evaluate an existing baseline checkpoint without using test data to select it."""

import argparse
import json
from pathlib import Path

import torch

from dust.data import read_manifest, write_json
from dust.evaluation import evaluate, select_frames, verify_records
from models.pointnet2_dust_msg import get_model
from train_dust import device_for, seed_everything, source_version


def main(args):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest = read_manifest(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["format_version"] not in (1, 2) or manifest["fingerprint"] != config["manifest_fingerprint"]:
        raise ValueError("Checkpoint and evaluation manifest differ")
    if config["is_smoke_run"] and not args.allow_smoke:
        raise ValueError("Smoke checkpoint: pass --allow-smoke for pipeline checks only")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("Frame limit must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output already exists; use a new directory")
    seed_everything(config["evaluation_seed"])
    device = device_for(args.device)
    model = get_model().to(device)
    model.load_state_dict(checkpoint["model_state"])
    records = select_frames(manifest, args.split, args.max_frames, config["evaluation_seed"])
    verify_records(args.data_root, records)
    report = evaluate(model, args.data_root, manifest, records, device,
                      threshold=config["threshold"], batch_size=config["eval_batch_size"],
                      seed=config["evaluation_seed"], output=output,
                      export_per_sensor=args.visualize_per_sensor,
                      backend=getattr(args, "backend", "eager"))
    report.update(split=args.split, checkpoint_epoch=checkpoint["epoch"] + 1,
                  checkpoint_source=config["source"], training_seed=config["seed"],
                  training_config=config, training_budget_epochs=checkpoint["training_budget_epochs"],
                  is_smoke_run=config["is_smoke_run"] or args.max_frames is not None,
                  resume_lineage=checkpoint.get("resume_lineage", []),
                  evaluation_source=source_version(),
                  checkpoint_selection="validation F1 at checkpoint's fixed threshold")
    write_json(output / "metrics.json", report)
    print(json.dumps(report["overall"], indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("eager", "cuda-graph"), default="eager")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--visualize-per-sensor", type=int, default=1)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
