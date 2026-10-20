"""Point/label alignment and shared preprocessing, independent of PyTorch."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from dust.grouping import CAPTURE_POLICY, assert_clean_capture_records

IGNORE = -100
KEEP_IDS = tuple(range(1, 9)) + tuple(range(12, 25)) + (26, 27, 28)
IGNORE_IDS = (0, 10, 11, 25)
LABEL_PROTOCOL = {"keep_ids": list(KEEP_IDS), "dust_ids": [9],
                  "ignore_ids": list(IGNORE_IDS), "ignore_index": IGNORE}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_labels(labels):
    if np.any(labels > 28):
        raise ValueError("Unexpected semantic label outside official IDs 0..28")
    target = np.full(labels.shape, IGNORE, dtype=np.int64)
    target[np.isin(labels, KEEP_IDS)] = 0
    target[labels == 9] = 1
    return target


def load_frame(root, record, verify_hash=False):
    """Return raw XYZI/labels; never reorder or independently filter labels."""
    root = Path(root).resolve()
    point_path = (root / record["path"]).resolve()
    label_path = (root / record["label_path"]).resolve()
    if root not in point_path.parents or root not in label_path.parents:
        raise ValueError("Frame paths must stay inside data root")
    point_bytes, label_bytes = point_path.stat().st_size, label_path.stat().st_size
    if point_bytes == 0 or point_bytes % 16 or point_bytes // 16 != label_bytes:
        raise ValueError(f"Point/label byte-count mismatch: {record['path']}")
    if "n_raw" in record and point_bytes // 16 != record["n_raw"]:
        raise ValueError(f"Frame changed since audit: {record['path']}")
    if verify_hash:
        for path, key in ((point_path, "point_sha256"), (label_path, "label_sha256")):
            if file_sha256(path) != record[key]:
                raise ValueError(f"Content changed since audit: {path}")
    points = np.fromfile(point_path, dtype="<f4").reshape(-1, 4)
    labels = np.fromfile(label_path, dtype=np.uint8)
    binary_labels(labels)  # Fail on unsupported IDs even at non-finite points.
    return points, labels


def finite_frame(points, labels):
    mask = np.isfinite(points).all(axis=1)
    index = np.flatnonzero(mask)
    return points[mask], binary_labels(labels[mask]), index


def validate_geometry(geometry):
    size, stride = geometry["block_size"], geometry["stride"]
    if not all(math.isfinite(v) and v > 0 for v in
               (size, stride, geometry["coord_scale"])):
        raise ValueError("Block size, stride and coordinate scale must be finite and positive")
    ratio = size / stride
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
        raise ValueError("block_size / stride must be a positive integer")
    if geometry["npoint"] < 1024:
        raise ValueError("The unchanged MSG architecture requires npoint >= 1024")


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = manifest.pop("fingerprint")
    if fingerprint(manifest) != digest:
        raise ValueError("Manifest fingerprint mismatch")
    manifest["fingerprint"] = digest
    if manifest["version"] != 1 or manifest["labels"] != LABEL_PROTOCOL:
        raise ValueError("Unsupported manifest/label protocol")
    validate_geometry(manifest["geometry"])
    if "preparation_protocol" in manifest:
        if manifest["preparation_protocol"] != CAPTURE_POLICY:
            raise ValueError("Unsupported preparation protocol")
        assert_clean_capture_records(manifest["frames"], manifest.get("group_map_sha256") is not None)
    seen_paths, group_splits = set(), {}
    for record in manifest["frames"]:
        if record["path"] in seen_paths or record["split"] not in ("train", "val", "test"):
            raise ValueError("Duplicate frame or invalid split")
        seen_paths.add(record["path"])
        group = record["group"]
        if group in group_splits and group_splits[group] != record["split"]:
            raise ValueError("A capture group leaks between splits")
        group_splits[group] = record["split"]
        if record["sensor"] not in manifest["intensity"]:
            raise ValueError("Sensor has no training intensity statistics")
    return manifest


def features(points, center, sensor_stats, geometry):
    out = points.copy().astype(np.float32)
    out[:, :3] = (out[:, :3] - np.asarray(center, dtype=np.float32)) / geometry["coord_scale"]
    logged = np.log1p(np.maximum(out[:, 3].astype(np.float64), 0))
    out[:, 3] = (logged - sensor_stats["mean"]) / sensor_stats["std"]
    if not np.isfinite(out).all():
        raise ValueError("Non-finite normalized features")
    return out


def evaluation_blocks(points, geometry, rng):
    """Occupied shifted grids: every finite point receives one vote per grid.

    Yield padded point indices, number of real points, and a shared tile center.
    Padding provides network context but must never contribute extra votes.
    """
    validate_geometry(geometry)
    size, stride, npoint = (geometry[k] for k in ("block_size", "stride", "npoint"))
    if len(points) == 0:
        return
    offsets = range(round(size / stride))
    for ix in offsets:
        for iy in offsets:
            offset = np.array([ix * stride, iy * stride], dtype=np.float64)
            cells = np.floor((points[:, :2].astype(np.float64) - offset) / size).astype(np.int64)
            order = np.lexsort((cells[:, 1], cells[:, 0]))
            sorted_cells = cells[order]
            boundaries = np.flatnonzero(np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)) + 1
            for tile in np.split(order, boundaries):
                cell = cells[tile[0]]
                center = np.array([*(offset + (cell + .5) * size),
                                   points[tile, 2].mean(dtype=np.float64)])
                tile = rng.permutation(tile)
                for start in range(0, len(tile), npoint):
                    real = tile[start:start + npoint]
                    count = len(real)
                    if count < npoint:
                        real = np.concatenate([real, rng.choice(tile, npoint - count, replace=True)])
                    yield real, count, center
