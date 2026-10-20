"""Descriptive server-side checks; these do not establish a causal failure mode."""

from pathlib import Path

import numpy as np

from dust.data import IGNORE, binary_labels, file_sha256


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "quantiles_0_10_50_90_100": None}
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "quantiles_0_10_50_90_100": np.quantile(values, [0., .1, .5, .9, 1.]).tolist()}


def point_distribution(points):
    return {"range": distribution(np.linalg.norm(points[:, :3].astype(np.float64), axis=1)),
            "z": distribution(points[:, 2]), "intensity": distribution(points[:, 3])}


def semantic_inventory(root, records):
    """Read only labels; subtract audited nonfinite histograms, without guessing a mask."""
    result = {}
    by_sequence = {}
    for index, record in enumerate(records, 1):
        path = Path(root) / record["label_path"]
        if file_sha256(path) != record["label_sha256"]:
            raise ValueError(f"Label content changed: {record['label_path']}")
        labels = np.fromfile(path, dtype=np.uint8)
        if len(labels) != record["n_raw"]:
            raise ValueError("Label count differs from audited frame")
        binary_labels(labels)
        finite = np.bincount(labels, minlength=29) - np.asarray(record["nonfinite_label_counts"])
        if np.any(finite < 0) or int(finite.sum()) != record["n_finite"]:
            raise ValueError("Audited finite label histogram is inconsistent")
        bucket = result.setdefault(record["split"], {}).setdefault(record["sensor"],
                     {"frames": 0, "finite_semantic_counts": np.zeros(29, dtype=np.int64),
                      "frames_with_car": 0})
        bucket["frames"] += 1
        bucket["finite_semantic_counts"] += finite
        bucket["frames_with_car"] += int(finite[12] > 0)
        row = by_sequence.setdefault(record["sequence"], {"sensor": record["sensor"],
                                 "split": record["split"], "frames": 0,
                                 "finite_semantic_counts": np.zeros(29, dtype=np.int64)})
        row["frames"] += 1
        row["finite_semantic_counts"] += finite
        if index % 500 == 0 or index == len(records):
            print(f"Verified label coverage {index}/{len(records)}", flush=True)
    for sensors in result.values():
        for bucket in sensors.values():
            bucket["finite_semantic_counts"] = bucket["finite_semantic_counts"].tolist()
    for row in by_sequence.values():
        row["finite_semantic_counts"] = row["finite_semantic_counts"].tolist()
    return {"by_split_sensor": result, "by_sequence": by_sequence,
            "scope": "finite-label counts based on verified label files and the saved audit's nonfinite counts"}


def select_review_rows(rows, sequence, limit):
    if limit < 1:
        raise ValueError("Need a positive review frame limit")
    rows = sorted((r for r in rows if r["sequence"] == sequence), key=lambda r: r["path"])
    if not rows:
        raise ValueError(f"Sequence absent from supplied evaluation results: {sequence}")
    ranked = sorted(rows, key=lambda r: (r["false_deletion_rate"] or 0., r["path"]), reverse=True)
    priorities = [ranked[0], rows[0], max(rows, key=lambda r: r["fp"]),
                  ranked[-1], ranked[len(ranked) // 2], max(rows, key=lambda r: r["fn"])] + ranked
    selected = {}
    for row in priorities:
        selected.setdefault(row["path"], row)
        if len(selected) == min(limit, len(rows)):
            break
    return list(selected.values())


def read_boxes(path):
    """Preserve the official h,w,l,x,y,z,angle fields; do not relabel any points."""
    if not Path(path).exists():
        return []
    boxes = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"Unexpected box annotation format: {path}")
        values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values[:3] <= 0):
            raise ValueError(f"Invalid box annotation: {path}")
        boxes.append(dict(zip(("class", "height", "width", "length", "x", "y", "z", "angle"),
                              [fields[0]] + values.tolist())))
    return boxes


def inside_boxes(xyz, boxes, category, origin="center"):
    """Diagnostic hypotheses: radians, Z yaw, X length/Y width, center or bottom Z."""
    if origin not in ("center", "bottom"):
        raise ValueError("Unknown box origin hypothesis")
    selected = np.zeros(len(xyz), dtype=bool)
    for box in boxes:
        if box["class"].lower() != category.lower():
            continue
        if abs(box["angle"]) > 2 * np.pi + 1e-6:
            raise ValueError("Box angle does not fit the diagnostic radians hypothesis")
        center = np.array([box[k] for k in ("x", "y", "z")])
        if origin == "bottom":
            center[2] += box["height"] / 2
        delta = xyz.astype(np.float64) - center
        cosine, sine = np.cos(box["angle"]), np.sin(box["angle"])
        x = delta[:, 0] * cosine + delta[:, 1] * sine
        y = -delta[:, 0] * sine + delta[:, 1] * cosine
        selected |= ((np.abs(x) <= box["length"] / 2) & (np.abs(y) <= box["width"] / 2)
                     & (np.abs(delta[:, 2]) <= box["height"] / 2))
    return selected


def duplicate_label_differences(root, records, duplicate_groups):
    """Quantify same-geometry annotation disagreement before quarantining it."""
    by_path = {r["path"]: r for r in records}
    comparisons = []
    for group in duplicate_groups:
        members = group["members"]
        reference = by_path[members[0]["path"]]
        raw_path = Path(root) / reference["path"]
        if file_sha256(raw_path) != reference["point_sha256"]:
            raise ValueError("Duplicate reference cloud changed")
        raw = np.fromfile(raw_path, dtype="<f4").reshape(-1, 4)
        finite = np.isfinite(raw).all(axis=1)
        labels = []
        for member in members:
            record = by_path[member["path"]]
            if file_sha256(Path(root) / record["path"]) != reference["point_sha256"]:
                raise ValueError("Duplicate cloud changed since audit")
            label_path = Path(root) / record["label_path"]
            if file_sha256(label_path) != record["label_sha256"]:
                raise ValueError("Duplicate annotation changed since audit")
            value = np.fromfile(label_path, dtype=np.uint8)
            if len(value) != len(raw):
                raise ValueError("Duplicate annotation length mismatch")
            labels.append(value[finite])
        first = labels[0]
        for member, other in zip(members[1:], labels[1:]):
            a, b = binary_labels(first), binary_labels(other)
            changed = first != other
            transitions = np.bincount((first[changed].astype(np.int64) * 29 + other[changed]), minlength=29*29)
            comparisons.append({"reference": members[0]["path"], "other": member["path"],
                                "finite_points": len(first), "semantic_disagreements": int(changed.sum()),
                                "binary_disagreements": int(np.sum(a != b)),
                                "keep_dust_contradictions": int(np.sum(((a == 0) & (b == 1)) |
                                                                      ((a == 1) & (b == 0)))),
                                "ignore_supervised_differences": int(np.sum((a == IGNORE) != (b == IGNORE))),
                                "transitions": {f"{i//29}->{i%29}": int(n) for i,n in enumerate(transitions) if n}})
    return comparisons
