"""Full-frame coverage, duplicate-free voting, metrics, and PLY exports."""

from collections import defaultdict
from pathlib import Path
from time import perf_counter
import zlib

import numpy as np
import torch

from dust.data import (IGNORE, KEEP_IDS, evaluation_blocks, features, finite_frame,
                       load_frame, write_json)
from dust.metrics import confusion, summarize
from dust.inference import InferenceForward


class EvaluationInterrupted(InterruptedError):
    """A cooperative stop requested between prediction batches."""


def select_frames(manifest, split, limit=None, seed=42):
    records = [r for r in manifest["frames"] if r["split"] == split]
    if limit is None:
        return records
    if limit < 1:
        raise ValueError("Frame limit must be positive")
    # Debug subsets sample across sensors, not the first lexicographic frames.
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for record in records:
        buckets[record["sensor"]].append(record)
    for sensor in buckets:
        rng.shuffle(buckets[sensor])
    selected = []
    while len(selected) < min(limit, len(records)):
        for sensor in sorted(buckets):
            if buckets[sensor] and len(selected) < limit:
                selected.append(buckets[sensor].pop())
    return sorted(selected, key=lambda r: r["path"])


def verify_records(root, records):
    for i, record in enumerate(records):
        load_frame(root, record, verify_hash=True)
        if i % 100 == 0 or i + 1 == len(records):
            print(f"Verified content hashes {i + 1}/{len(records)}", flush=True)


def predict_frame(model, points, record, manifest, device, batch_size=4, seed=42,
                  should_stop=None, forward=None):
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    sums = np.zeros(len(points), dtype=np.float64)
    votes = np.zeros(len(points), dtype=np.int64)
    frame_seed = (seed + zlib.crc32(record["path"].encode())) % (2**32)
    rng = np.random.default_rng(frame_seed)
    pending = []

    def flush():
        if not pending:
            return
        if should_stop is not None and should_stop():
            raise EvaluationInterrupted("Validation interrupted; retry this validation on resume")
        array = np.stack([p[0] for p in pending])
        tensor = torch.from_numpy(array).transpose(1, 2).to(device)
        log_probability, _ = (model if forward is None else forward)(tensor)
        if not torch.isfinite(log_probability).all():
            raise RuntimeError("Non-finite model prediction, including ignored context points")
        scores = log_probability[:, :, 1].exp().cpu().numpy()
        for score, (_, indices, real_count) in zip(scores, pending):
            real = indices[:real_count]
            # Real indices are unique within a block; padding is excluded.
            sums[real] += score[:real_count]
            votes[real] += 1
        pending.clear()

    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Upstream FPS starts at a random point. Freeze evaluation randomness per
    # frame and restore training RNG state after evaluating validation frames.
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.set_rng_state(torch.Generator().manual_seed(frame_seed).get_state())
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(frame_seed)
        for indices, count, center in evaluation_blocks(points, manifest["geometry"], rng):
            block = features(points[indices], center,
                             manifest["intensity"][record["sensor"]], manifest["geometry"])
            pending.append((block, indices, count))
            if len(pending) == batch_size:
                flush()
        flush()
    expected_votes = round(manifest["geometry"]["block_size"] / manifest["geometry"]["stride"])**2
    if np.any(votes != expected_votes):
        raise RuntimeError("Incomplete point coverage or duplicate padding votes")
    return sums / np.maximum(votes, 1), votes


def write_ply(path, xyz, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [(k, "<f4") for k in ("x", "y", "z")] + [(k, "u1") for k in ("red", "green", "blue")]
    vertices = np.empty(len(xyz), dtype=dtype)
    for i, k in enumerate(("x", "y", "z")):
        vertices[k] = xyz[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        vertices[k] = colors[:, i]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def export_frame(directory, record, raw, labels, finite_index, target, scores, threshold):
    prefix = Path(directory) / Path(record["path"]).with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    points = raw[finite_index]
    pred = scores >= threshold
    probabilities = np.full(len(raw), np.nan, dtype=np.float32)
    probabilities[finite_index] = scores
    predictions = np.full(len(raw), IGNORE, dtype=np.int16)
    predictions[finite_index] = pred.astype(np.int16)
    np.savez_compressed(str(prefix) + "_prediction.npz", dust_probability=probabilities,
                        prediction=predictions, semantic_label=labels, finite_index=finite_index,
                        threshold=np.array(threshold))
    # Filtering never uses true labels, including ignored semantic categories.
    points[~pred].astype("<f4").tofile(str(prefix) + "_filtered.bin")
    gt = np.tile([70, 180, 100], (len(points), 1)).astype(np.uint8)
    gt[target == 1] = [60, 120, 240]
    gt[target == IGNORE] = [150, 150, 150]
    predicted = np.tile([70, 180, 100], (len(points), 1)).astype(np.uint8)
    predicted[pred] = [60, 120, 240]
    error = gt.copy()
    error[(target == 0) & pred] = [255, 40, 40]
    error[(target == 1) & ~pred] = [255, 170, 20]
    write_ply(str(prefix) + "_ground_truth.ply", points[:, :3], gt)
    write_ply(str(prefix) + "_prediction.ply", points[:, :3], predicted)
    write_ply(str(prefix) + "_errors.ply", points[:, :3], error)
    write_ply(str(prefix) + "_filtered.ply", points[~pred, :3], predicted[~pred])


def evaluate(model, root, manifest, records, device, threshold=.5, batch_size=4,
             seed=42, output=None, export_per_sensor=0, should_stop=None, backend="eager"):
    if not records or not 0 < threshold < 1 or export_per_sensor < 0:
        raise ValueError("Need frames, 0 < threshold < 1 and nonnegative export count")
    model.eval()
    forward = InferenceForward(model, device, batch_size, backend)
    clock = perf_counter()
    total = np.zeros(4, dtype=np.int64)
    no_dust = np.zeros(4, dtype=np.int64)
    by_sensor = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    by_sequence = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    by_class = {str(k): [0, 0] for k in KEEP_IDS}  # total true points, deleted
    exported = defaultdict(int)
    rows = []
    for i, record in enumerate(records):
        frame_clock = perf_counter()
        if should_stop is not None and should_stop():
            raise EvaluationInterrupted("Validation interrupted")
        raw, labels = load_frame(root, record)
        points, target, original_index = finite_frame(raw, labels)
        probability, votes = predict_frame(model, points, record, manifest, device, batch_size, seed,
                                           should_stop=should_stop, forward=forward)
        counts = confusion(target, probability, threshold)
        total += counts
        by_sensor[record["sensor"]] += counts
        by_sequence[record["sequence"]] += counts
        if not np.any(target == 1):
            no_dust += counts
        semantic = labels[original_index]
        for label in KEEP_IDS:
            mask = semantic == label
            by_class[str(label)][0] += int(mask.sum())
            by_class[str(label)][1] += int(np.sum(mask & (probability >= threshold)))
        rows.append({"path": record["path"], "sensor": record["sensor"],
                     "sequence": record["sequence"], "group": record["group"],
                     "n_raw": len(raw), "n_finite": len(points),
                     "n_nonfinite": len(raw) - len(points),
                     "n_ignored_finite": int(np.sum(target == IGNORE)),
                     "min_votes": int(votes.min()) if len(votes) else 0,
                     "max_votes": int(votes.max()) if len(votes) else 0,
                     **summarize(counts)})
        if output is not None and exported[record["sensor"]] < export_per_sensor:
            export_frame(Path(output) / "visualizations", record, raw, labels,
                         original_index, target, probability, threshold)
            exported[record["sensor"]] += 1
        elapsed = perf_counter() - clock
        eta = elapsed / (i + 1) * (len(records) - i - 1) / 60
        print(f"Evaluated {i + 1}/{len(records)} {record['path']} F1={rows[-1]['f1']} "
              f"s/frame={perf_counter() - frame_clock:.2f} validation_ETA_min={eta:.1f}", flush=True)
    report = {"manifest_fingerprint": manifest["fingerprint"], "threshold": threshold,
              "evaluation_seed": seed, "evaluation_batch_size": batch_size,
              "inference_backend": backend, "inference": forward.statistics(),
              "evaluation_seconds": perf_counter() - clock,
              "frames": len(records), "overall": summarize(total),
              "by_sensor": {k: summarize(v) for k, v in sorted(by_sensor.items())},
              "by_sequence": {k: summarize(v) for k, v in sorted(by_sequence.items())},
              "no_dust_frames": summarize(no_dust),
              "false_deletion_by_semantic_class": {
                  k: {"n_valid": v[0], "n_deleted": v[1],
                      "false_deletion_rate": v[1] / v[0] if v[0] else None}
                  for k, v in by_class.items()},
              "point_coverage": "all finite points; one vote per shifted grid; padding excluded",
              "undefined_metrics": "null when denominator is zero"}
    if output is not None:
        write_json(Path(output) / "metrics.json", report)
        write_json(Path(output) / "frames.json", rows)
    return report
