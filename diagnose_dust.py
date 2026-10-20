"""Inspect legacy failure cases on the server without training or tuning on test."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from dust.data import (file_sha256, fingerprint, finite_frame, load_frame, read_manifest, write_json)
from dust.diagnostics import (distribution, duplicate_label_differences, inside_boxes,
                              point_distribution, read_boxes, select_review_rows, semantic_inventory)
from dust.evaluation import export_frame, predict_frame, write_ply
from dust.grouping import repair_capture_records, write_capture_reports
from dust.inference import InferenceForward
from dust.metrics import confusion
from dust.semantics import SEMANTIC_NAMES, SEMANTIC_SOURCE
from models.pointnet2_dust_msg import get_model
from train_dust import device_for, seed_everything, source_version


def inspect_case(root, output, record, reference, manifest, report, semantic_id, model=None, forward=None,
                 device=None, existing_visualizations=None):
    raw, labels = load_frame(root, record, verify_hash=True)
    points, target, finite_index = finite_frame(raw, labels)
    threshold = report["threshold"]
    if model is not None:
        probability, votes = predict_frame(model, points, record, manifest, device,
            report["evaluation_batch_size"], report["evaluation_seed"], forward=forward)
        prediction = probability >= threshold
        export_frame(output / "visualizations", record, raw, labels, finite_index,
                     target, probability, threshold)
        origin = "diagnostic forward using the fixed legacy validation-selected checkpoint"
        rounding = 0
    else:
        path = Path(existing_visualizations) / Path(record["path"]).with_suffix("")
        path = Path(str(path) + "_prediction.npz")
        if not path.exists():
            return {"path": record["path"], "prediction_available": False,
                    "reason": "No exported per-point prediction; supply --checkpoint for this review case"}
        with np.load(path, allow_pickle=False) as saved:
            if (not np.array_equal(saved["semantic_label"], labels)
                    or not np.array_equal(saved["finite_index"], finite_index)
                    or float(saved["threshold"]) != threshold):
                raise ValueError("Exported labels/indices/threshold differ from the audited raw frame")
            probability = saved["dust_probability"][finite_index]
            prediction = saved["prediction"][finite_index]
            if not np.isin(prediction, [0, 1]).all():
                raise ValueError("Invalid exported prediction")
            prediction = prediction.astype(bool)
            rounding = int(np.sum(prediction != (probability >= threshold)))
        votes = None
        origin = "saved export; stored binary prediction retains pre-float32-rounding decisions"
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid review probabilities")
    # Binary predictions are authoritative for old float32 NPZ exports.
    counts = confusion(target, prediction.astype(np.float64), .5)
    expected = [reference[k] for k in ("tn", "fp", "fn", "tp")]
    semantic = labels[finite_index]
    selected, dust = semantic == semantic_id, semantic == 9
    deleted, kept = selected & prediction, selected & ~prediction
    boxes = read_boxes((Path(root) / record["path"]).with_suffix(".txt"))
    box_overlap = {}
    for origin_hypothesis in ("center", "bottom"):
        box_overlap[origin_hypothesis] = {}
        for category in ("Car", "Dust"):
            mask = inside_boxes(points[:, :3], boxes, category, origin_hypothesis)
            box_overlap[origin_hypothesis][category] = {
                "n_boxes": sum(b["class"].lower() == category.lower() for b in boxes),
                "selected_label_points_inside": int(np.sum(selected & mask)),
                "selected_deleted_points_inside": int(np.sum(deleted & mask))}
    colors = np.tile([100, 100, 100], (len(points), 1)).astype(np.uint8)
    colors[dust] = [60, 120, 240]
    colors[kept] = [70, 180, 100]
    colors[deleted] = [255, 40, 40]
    prefix = output / "review" / Path(record["path"]).with_suffix("")
    write_ply(str(prefix) + "_car_review.ply", points[:, :3], colors)
    write_json(str(prefix) + "_boxes.json", {"boxes": boxes, "overlap_hypotheses": box_overlap,
               "assumptions": "angle radians; Z yaw; X length/Y width; both center-Z and bottom-Z shown. "
                              "These hypotheses are not label corrections or proof of annotation quality."})
    ranges = np.linalg.norm(points[:, :3].astype(np.float64), axis=1)
    range_rows = []
    for low, high in ((0, 10), (10, 20), (20, 40), (40, 80), (80, np.inf)):
        mask = selected & (ranges >= low) & (ranges < high)
        n = int(mask.sum())
        range_rows.append({"range_min": low, "range_max": None if np.isinf(high) else high,
                           "n_selected": n, "n_deleted": int(np.sum(mask & prediction)),
                           "deletion_rate": float(np.mean(prediction[mask])) if n else None})
    return {"path": record["path"], "prediction_available": True, "prediction_origin": origin,
            "raw_hashes_verified": True, "shared_finite_index_verified": True,
            "confusion_matches_saved_frame": counts.tolist() == expected,
            "counts": counts.tolist(), "saved_counts": expected,
            "float32_probability_threshold_disagreements": rounding,
            "votes_min": int(votes.min()) if votes is not None and len(votes) else None,
            "votes_max": int(votes.max()) if votes is not None and len(votes) else None,
            "selected_label": semantic_id, "selected_label_name": SEMANTIC_NAMES[semantic_id],
            "selected_points": int(selected.sum()), "selected_deleted": int(deleted.sum()),
            "selected_deletion_rate": float(np.mean(prediction[selected])) if selected.any() else None,
            "selected_deleted_geometry": point_distribution(points[deleted]),
            "selected_retained_geometry": point_distribution(points[kept]),
            "dust_geometry": point_distribution(points[dust]),
            "selected_dust_probability": distribution(probability[selected]),
            "dust_dust_probability": distribution(probability[dust]),
            "deletion_by_range": range_rows, "box_overlap_hypotheses": box_overlap}


def main(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root, output = Path(args.data_root).resolve(), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output already exists; use a new directory")
    if args.semantic_id != 12:
        raise ValueError("This focused audit is defined for official Car ID 12")
    manifest = read_manifest(args.manifest)
    results = Path(args.results)
    report = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    frames = json.loads((results / "frames.json").read_text(encoding="utf-8"))
    if report["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Legacy result and manifest differ")
    selected = select_review_rows(frames, args.sequence, args.review_frames)
    records = {r["path"]: r for r in manifest["frames"]}
    if any(row["path"] not in records for row in selected):
        raise ValueError("Review frame absent from manifest")
    output.mkdir(parents=True, exist_ok=True)
    _, capture_report = repair_capture_records(manifest["frames"])
    write_capture_reports(output, capture_report)
    print("Inspecting label coverage across splits/sensors...", flush=True)
    inventory = semantic_inventory(root, manifest["frames"])
    write_json(output / "label_inventory.json", inventory)
    print("Comparing duplicate annotation copies...", flush=True)
    disagreements = duplicate_label_differences(root, manifest["frames"], capture_report["duplicates"])
    write_json(output / "duplicate_label_differences.json", disagreements)
    model = forward = device = None
    checkpoint_hash = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = state["config"]
        checkpoint_hash = file_sha256(args.checkpoint)
        if (state["format_version"] not in (1, 2) or config["manifest_fingerprint"] != manifest["fingerprint"]
                or state["epoch"] + 1 != report["checkpoint_epoch"]
                or config["source"] != report["checkpoint_source"]
                or config["seed"] != report["training_seed"]
                or config["threshold"] != report["threshold"]
                or config["eval_batch_size"] != report["evaluation_batch_size"]
                or config["evaluation_seed"] != report["evaluation_seed"]
                or (report.get("checkpoint_sha256") is not None and checkpoint_hash != report["checkpoint_sha256"])):
            raise ValueError("Diagnostic checkpoint does not match saved evaluation identity/settings")
        seed_everything(config["evaluation_seed"])
        device = device_for(args.device)
        model = get_model().to(device).eval()
        model.load_state_dict(state["model_state"])
        forward = InferenceForward(model, device, config["eval_batch_size"], args.backend)
    cases = []
    for index, reference in enumerate(selected):
        case = inspect_case(root, output, records[reference["path"]], reference, manifest, report,
                            args.semantic_id, model, forward, device, results / "visualizations")
        cases.append(case)
        print(f"Reviewed {index + 1}/{len(selected)} {reference['path']} "
              f"Car deletion={case.get('selected_deletion_rate')}", flush=True)
    result = {"is_diagnostic": True, "source": source_version(), "legacy_manifest": manifest["fingerprint"],
              "input_fingerprint": fingerprint([{k: r[k] for k in ("path", "point_sha256", "label_sha256")}
                                                for r in sorted(manifest["frames"], key=lambda r: r["path"])]),
              "legacy_results_split": report["split"], "checkpoint_sha256": checkpoint_hash,
              "official_semantic_source": SEMANTIC_SOURCE, "mapping": {"12": "Car -> keep", "9": "Dust -> dust"},
              "sequence": args.sequence, "saved_sequence_metrics": report.get("by_sequence", {}).get(args.sequence),
              "saved_car_metrics": report["false_deletion_by_semantic_class"].get("12"),
              "capture_summary": {k: capture_report[k] for k in ("input_frames", "retained_frames", "excluded_frames",
                                                    "duplicate_point_groups", "conflicting_label_groups", "exclusion_reason_counts")},
              "duplicate_binary_contradiction_pairs": sum(r["keep_dust_contradictions"] > 0 for r in disagreements),
              "label_inventory_file": "label_inventory.json", "cases": cases,
              "all_review_predictions_available": all(c["prediction_available"] for c in cases),
              "all_review_confusions_match": all(c.get("confusion_matches_saved_frame", False) for c in cases),
              "interpretation_limits": "Mapping and storage/index checks can rule out specific implementation errors. "
                  "Range/intensity/bbox overlap statistics are descriptive, not causal proof. Manual label/shape review "
                  "and later train/validation-only ablations are required before attributing failure to structure loss. "
                  "Legacy test is used for forensic diagnosis, not threshold/hyperparameter selection."}
    write_json(output / "summary.json", result)
    (output / "review_frames.txt").write_text("".join(row["path"] + "\n" for row in selected), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("sequence", "capture_summary", "all_review_predictions_available",
                                           "all_review_confusions_match")}), flush=True)
    if model is not None and not result["all_review_confusions_match"]:
        raise RuntimeError("Diagnostic predictions differ from saved evaluation; inspect summary before proceeding")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True, help="Existing full evaluation directory")
    parser.add_argument("--checkpoint", help="Optional fixed legacy best.pt to review frames without saved NPZs")
    parser.add_argument("--sequence", default="sequence_0114")
    parser.add_argument("--semantic-id", type=int, default=12)
    parser.add_argument("--review-frames", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("eager", "cuda-graph"), default="cuda-graph")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
