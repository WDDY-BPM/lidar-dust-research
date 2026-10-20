"""Audit LiDARDustX and freeze group splits/statistics on the experiment server."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np

from dust.data import (LABEL_PROTOCOL, binary_labels, file_sha256, fingerprint,
                       load_frame, validate_geometry, write_json)
from dust.grouping import CAPTURE_POLICY, repair_capture_records, write_capture_reports


def assign_splits(records, seed, ratios):
    """Stratify capture groups by sensor composition, never individual frames."""
    groups = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)
    strata = defaultdict(list)
    for name, frames in groups.items():
        strata[tuple(sorted({r["sensor"] for r in frames}))].append(name)
    rng = np.random.default_rng(seed)
    result, assigned = {}, np.zeros(3, dtype=np.int64)
    for signature in sorted(strata):
        names = list(rng.permutation(sorted(strata[signature])))
        n = len(names)
        if n >= 3:
            n_train = min(n - 2, max(1, round(n * ratios[0])))
            n_val = min(n - n_train - 1, max(1, round(n * ratios[1])))
            slots = [0] * n_train + [1] * n_val + [2] * (n - n_train - n_val)
        else:
            slots = []
            for _ in names:
                slot = int(np.argmin((assigned + 1) / np.asarray(ratios)))
                slots.append(slot)
                assigned[slot] += 1
        for name, slot in zip(names, slots):
            result[name] = ("train", "val", "test")[slot]
        if n >= 3:
            assigned += np.bincount(slots, minlength=3)
    return result


def group_mapping(path):
    if path is None:
        return None
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sequence, group = row["sequence"].strip(), row["group"].strip()
            if not sequence or not group or sequence in result:
                raise ValueError("Group CSV requires unique nonempty sequence,group rows")
            result[sequence] = group
    return result


def prepare(args):
    root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output must be empty; use a new directory for a new protocol")
    geometry = {"npoint": args.npoint, "block_size": args.block_size,
                "stride": args.stride, "coord_scale": args.coord_scale}
    validate_geometry(geometry)
    ratios = (args.train_ratio, args.val_ratio, 1 - args.train_ratio - args.val_ratio)
    if any(not np.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("Train, val and test ratios must all be positive")
    mapping = group_mapping(args.group_map)
    paths = sorted(root.rglob("*.bin"))
    if not paths:
        raise ValueError("No .bin frames found under data root")
    paired_labels = {p.with_suffix(".label") for p in paths}
    orphan_labels = set(root.rglob("*.label")) - paired_labels
    if orphan_labels:
        raise ValueError(f"Orphan labels: {sorted(map(str, orphan_labels))[:5]}")
    records, histograms = [], defaultdict(lambda: np.zeros(29, dtype=np.int64))
    for i, path in enumerate(paths):
        relative = path.relative_to(root)
        match = re.match(r"^(sequence_\d+)_([^_]+)_", path.stem)
        if match is None or len(relative.parts) < 2 or relative.parts[0] != match[2]:
            raise ValueError(f"Unsupported sensor/frame naming: {relative}")
        sequence, sensor = match[1], match[2]
        if mapping is not None and sequence not in mapping:
            raise ValueError(f"Sequence missing from group map: {sequence}")
        record = {"path": relative.as_posix(),
                  "label_path": relative.with_suffix(".label").as_posix(),
                  "sensor": sensor, "sequence": sequence,
                  "group": mapping[sequence] if mapping is not None else sequence}
        points, labels = load_frame(root, record)
        finite = np.isfinite(points).all(axis=1)
        target = binary_labels(labels[finite])
        counts = [int(np.sum(target == c)) for c in (0, 1)]
        record.update(n_raw=len(labels), n_finite=int(finite.sum()),
                      n_nonfinite=int((~finite).sum()),
                      n_ignored_finite=int(np.sum(target == -100)), class_counts=counts,
                      finite_semantic_counts=np.bincount(labels[finite], minlength=29).tolist(),
                      n_negative_intensity=int(np.sum(points[finite, 3] < 0)),
                      nonfinite_label_counts=np.bincount(labels[~finite], minlength=29).tolist(),
                      point_sha256=file_sha256(path),
                      label_sha256=file_sha256(path.with_suffix(".label")))
        histograms[sensor] += np.bincount(labels, minlength=29)
        records.append(record)
        if i % 100 == 0 or i + 1 == len(paths):
            print(f"Audited {i + 1}/{len(paths)} frames", flush=True)
    if mapping is not None and set(mapping) != {r["sequence"] for r in records}:
        raise ValueError("Group map must describe exactly the dataset's sequences")
    input_frames = len(records)
    input_fingerprint = fingerprint([{k: r[k] for k in ("path", "point_sha256", "label_sha256")}
                                     for r in records])
    records, capture_report = repair_capture_records(records, mapping)
    splits = assign_splits(records, args.seed, ratios)
    for record in records:
        record["split"] = splits[record["group"]]
    sensors = sorted(histograms)
    split_report = {}
    for split in ("train", "val", "test"):
        selected = [r for r in records if r["split"] == split]
        if {r["sensor"] for r in selected} != set(sensors):
            raise ValueError(f"{split} lacks a sensor; review capture groups and split seed")
        class_counts = np.sum([r["class_counts"] for r in selected], axis=0)
        if np.any(class_counts == 0):
            raise ValueError(f"{split} must contain both supervised classes")
        split_report[split] = {"frames": len(selected),
                               "groups": len({r["group"] for r in selected}),
                               "class_counts": class_counts.tolist(),
                               "no_dust_frames": sum(r["class_counts"][1] == 0 for r in selected),
                               "sensor_frames": {s: sum(r["sensor"] == s for r in selected) for s in sensors}}
    # Log-intensity moments come exclusively from finite training points.
    moments = {s: [0, 0., 0.] for s in sensors}
    for record in records:
        if record["split"] != "train":
            continue
        points, _ = load_frame(root, record)
        selected = points[np.isfinite(points).all(axis=1), 3].astype(np.float64)
        values = np.log1p(np.maximum(selected, 0))
        moment = moments[record["sensor"]]
        moment[0] += len(values)
        moment[1] += float(values.sum())
        moment[2] += float(np.square(values).sum())
    intensity = {}
    for sensor, (count, total, squared) in moments.items():
        if count == 0:
            raise ValueError(f"No finite training points for {sensor}")
        mean = total / count
        intensity[sensor] = {"mean": mean,
                             "std": max(float(np.sqrt(max(squared / count - mean**2, 0))), 1e-6),
                             "n_train_finite": count}
    counts = np.array(split_report["train"]["class_counts"], dtype=np.float64)
    weights = 1 / np.sqrt(counts)
    weights /= weights.mean()
    manifest = {"version": 1, "dataset": "LiDARDustX", "split_seed": args.seed,
                "split_ratios_by_group": list(ratios), "labels": LABEL_PROTOCOL,
                "grouping": "capture_day_components_with_user_map" if mapping is not None else "capture_day_components",
                "preparation_protocol": CAPTURE_POLICY,
                "input_fingerprint": input_fingerprint,
                "group_map_sha256": file_sha256(args.group_map) if mapping is not None else None,
                "geometry": geometry, "intensity_transform": "log1p(max(intensity,0)); training_sensor_zscore",
                "intensity": intensity, "class_weights": weights.tolist(), "frames": records}
    manifest["fingerprint"] = fingerprint(manifest)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_capture_reports(output, capture_report)
    write_json(output / "audit.json", {"frames": len(records), "splits": split_report,
               "input_frames": input_frames, "excluded_frames": capture_report["excluded_frames"],
               "preparation_protocol": CAPTURE_POLICY,
               "duplicate_point_groups": capture_report["duplicate_point_groups"],
               "conflicting_label_groups": capture_report["conflicting_label_groups"],
               "exclusion_reason_counts": capture_report["exclusion_reason_counts"],
               "finite_semantic_histogram_by_split_sensor": {
                   split: {sensor: np.sum([r["finite_semantic_counts"] for r in records
                                           if r["split"] == split and r["sensor"] == sensor], axis=0).tolist()
                           for sensor in sensors} for split in ("train", "val", "test")},
               "raw_label_histogram_by_sensor": {s: h.tolist() for s, h in histograms.items()},
               "nonfinite_points": sum(r["n_nonfinite"] for r in records),
               "label_alignment": "count and shared index verified; semantic alignment requires visual review",
               "grouping_caveat": CAPTURE_POLICY["claim"]})
    for split in ("train", "val", "test"):
        (output / f"{split}.txt").write_text("".join(r["path"] + "\n" for r in records
                                                    if r["split"] == split), encoding="utf-8")
    print(json.dumps(split_report, indent=2), flush=True)
    print(f"Clean capture policy: retained {len(records)}/{input_frames}; "
          f"excluded {capture_report['excluded_frames']}. {CAPTURE_POLICY['claim']}", flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="artifacts/lidardustx")
    parser.add_argument("--group-map", help="CSV with sequence,group; cover every sequence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=.7)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--npoint", type=int, default=4096)
    parser.add_argument("--block-size", type=float, default=10.)
    parser.add_argument("--stride", type=float, default=5.)
    parser.add_argument("--coord-scale", type=float, default=10.)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
