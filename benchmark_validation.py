"""Server-only comparison of eager/CUDA Graph full-frame validation, same weights."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dust.data import file_sha256, finite_frame, load_frame, read_manifest, write_json
from dust.evaluation import predict_frame, select_frames, verify_records
from dust.inference import InferenceForward
from dust.metrics import confusion, summarize
from models.pointnet2_dust_msg import get_model
from train_dust import device_for, seed_everything, source_version


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Benchmark output already exists; use a new path")
    if args.max_frames < 1:
        raise ValueError("Need at least one validation frame")
    manifest = read_manifest(args.manifest)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint and manifest differ")
    device = device_for(args.device)
    if device.type != "cuda":
        raise ValueError("Run the CUDA Graph comparison on the designated GPU server")
    seed_everything(config["evaluation_seed"])
    model = get_model().to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    records = select_frames(manifest, "val", args.max_frames, config["evaluation_seed"])
    verify_records(args.data_root, records)
    runners = {backend: InferenceForward(model, device, config["eval_batch_size"], backend)
               for backend in ("eager", "cuda-graph")}
    totals = {backend: 0. for backend in runners}
    rows = []
    for index, record in enumerate(records):
        raw, labels = load_frame(args.data_root, record)
        points, target, _ = finite_frame(raw, labels)
        predictions, timing = {}, {}
        cpu_before = torch.get_rng_state().clone()
        gpu_before = torch.cuda.get_rng_state(device).clone()
        for backend, forward in runners.items():
            torch.cuda.synchronize(device)
            clock = perf_counter()
            predictions[backend] = predict_frame(model, points, record, manifest, device,
                config["eval_batch_size"], config["evaluation_seed"], forward=forward)
            torch.cuda.synchronize(device)
            timing[backend] = perf_counter() - clock
            totals[backend] += timing[backend]
            if (not torch.equal(cpu_before, torch.get_rng_state()) or
                    not torch.equal(gpu_before, torch.cuda.get_rng_state(device))):
                raise RuntimeError("Validation changed caller RNG state")
        eager, eager_votes = predictions["eager"]
        graph, graph_votes = predictions["cuda-graph"]
        counts = {backend: confusion(target, prediction[0], config["threshold"])
                  for backend, prediction in predictions.items()}
        max_error = float(np.max(np.abs(eager - graph), initial=0.))
        row = {"path": record["path"], "sensor": record["sensor"], "seconds": timing,
               "max_probability_error": max_error,
               "votes_equal": bool(np.array_equal(eager_votes, graph_votes)),
               "threshold_predictions_equal": bool(np.array_equal(eager >= config["threshold"],
                                                                   graph >= config["threshold"])),
               "confusion_equal": bool(np.array_equal(counts["eager"], counts["cuda-graph"])),
               "eager": summarize(counts["eager"]), "cuda_graph": summarize(counts["cuda-graph"])}
        rows.append(row)
        print(f"Compared {index + 1}/{len(records)} " + json.dumps(row), flush=True)
    compatible = runners["cuda-graph"].replays > 0 and all(row["votes_equal"] and row["threshold_predictions_equal"] and
                     row["confusion_equal"] and row["max_probability_error"] <= 1e-6 for row in rows)
    result = {"is_benchmark": True, "split": "val", "frames": len(rows),
              "source": source_version(), "checkpoint_sha256": file_sha256(args.checkpoint),
              "checkpoint_epoch": checkpoint["epoch"] + 1,
              "manifest_fingerprint": manifest["fingerprint"],
              "evaluation_batch_size": config["eval_batch_size"],
              "seconds": totals, "speedup_including_capture": totals["eager"] / totals["cuda-graph"],
              "inference": {key: runner.statistics() for key, runner in runners.items()},
              "sample_parity_passed": compatible, "rows": rows,
              "scope": "Same selected validation frames/weights. Sample parity is not proof for all frames; "
                       "timing includes first graph capture, excludes shared file loading/hash verification."}
    write_json(output, result)
    print(json.dumps({key: result[key] for key in ("frames", "seconds", "speedup_including_capture",
                                                 "sample_parity_passed")}), flush=True)
    if not compatible:
        raise RuntimeError("Parity comparison failed; keep eager evaluation and inspect this report")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/validation_benchmark.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
