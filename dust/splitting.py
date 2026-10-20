"""Allocate indivisible captures using audited counts, never model predictions."""

from collections import defaultdict
import hashlib

import numpy as np


SPLITS = ("train", "val", "test")
SPLIT_POLICY = {
    "version": 3,
    "name": "capture_coverage_balanced_v3",
    "unit": "complete capture group; no frame or sequence may be separated",
    "binary_minimum": "per sensor/split/class: min(1000, max(1, total_points // 100))",
    "car_minimum": "per sensor/split: min(100, max(1, total_car_points // 100)); "
                   "required only with Car in at least 3 independent capture groups",
    "objective": "normalized absolute deviations from requested frame/point shares; "
                 "global frames weight 4, sensor frames 1, global binary points 1, "
                 "sensor binary points 1, eligible sensor Car points 0.25",
    "tie_break": "SHA256(seed, group, split), total weight below 1e-8",
    "solver": "scipy==1.14.1 optimize.milp; gap 1e-6; time limit 30 seconds; node limit 100000",
    "scope": "label counts for stratification only; no predictions, F1 or old split assignments",
}


def _inventory(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    names, sensors = sorted(grouped), sorted({r["sensor"] for r in records})
    if not names:
        raise ValueError("Cannot split an empty collection of capture groups")
    frames = np.zeros((len(names), len(sensors)), dtype=np.int64)
    counts = np.zeros((len(names), len(sensors), 3), dtype=np.int64)
    for gi, group in enumerate(names):
        for record in grouped[group]:
            si = sensors.index(record["sensor"])
            binary = np.asarray(record["class_counts"], dtype=np.int64)
            semantic = np.asarray(record["finite_semantic_counts"], dtype=np.int64)
            if binary.shape != (2,) or semantic.shape != (29,) or np.any(binary < 0) or np.any(semantic < 0):
                raise ValueError("Split allocation requires audited nonnegative binary/semantic counts")
            if semantic[9] != binary[1] or semantic[12] > binary[0]:
                raise ValueError("Split semantic counts disagree with binary counts")
            frames[gi, si] += 1
            counts[gi, si] += [int(binary[0]), int(binary[1]), int(semantic[12])]
    return names, sensors, frames, counts


def _requirements(sensors, counts):
    requirements, limited = [], []
    for si, sensor in enumerate(sensors):
        for ci, category in enumerate(("keep", "dust", "Car")):
            values = counts[:, si, ci]
            total, support = int(values.sum()), int(np.count_nonzero(values))
            floor = min(100 if ci == 2 else 1000, max(1, total // 100))
            item = {"sensor": sensor, "category": category, "minimum_points_per_split": floor,
                    "independent_groups_with_points": support, "total_points": total}
            if ci == 2 and support < 3:
                limited.append({**item, "reason": "fewer than 3 independent Car captures; "
                                "three-way Car coverage cannot be required; report missing cells as unavailable"})
                continue
            if support < 3 or int(np.minimum(values, floor).sum()) < 3 * floor:
                raise ValueError(f"Insufficient independent {sensor}/{category} coverage for three splits; "
                                 "review capture metadata, do not split captures or weaken constraints silently")
            requirements.append((item, values))
    return requirements, limited


def split_quality(records):
    """Validate existing assignments without allocating or changing any frame."""
    names, sensors, frames, counts = _inventory(records)
    requirements, limited = _requirements(sensors, counts)
    assigned = {}
    for record in records:
        split = record.get("split")
        if split not in SPLITS or assigned.setdefault(record["group"], split) != split:
            raise ValueError("Capture group has an invalid or leaking split")
    slots = np.array([SPLITS.index(assigned[name]) for name in names])
    constraints = []
    for item, values in requirements:
        observed = {split: int(values[slots == slot].sum()) for slot, split in enumerate(SPLITS)}
        if any(n < item["minimum_points_per_split"] for n in observed.values()):
            raise ValueError(f"Insufficient split coverage for {item['sensor']}/{item['category']}: {observed}")
        constraints.append({**item, "points_by_split": observed})
    report = {}
    for slot, split in enumerate(SPLITS):
        selected = slots == slot
        n = int(frames[selected].sum())
        report[split] = {"frames": n, "frame_fraction": n / int(frames.sum()),
                         "groups": int(selected.sum()), "sensors": {
                             sensor: {"frames": int(frames[selected, si].sum()),
                                      "keep_points": int(counts[selected, si, 0].sum()),
                                      "dust_points": int(counts[selected, si, 1].sum()),
                                      "car_points": int(counts[selected, si, 2].sum())}
                             for si, sensor in enumerate(sensors)}}
    return {"policy": SPLIT_POLICY, "coverage_passed": True, "constraints": constraints,
            "limited_car_coverage": limited, "splits": report}


def allocate_splits(records, seed, ratios):
    """Solve a fixed count-only MILP; fail rather than fall back to weak coverage.

    SciPy's documented deterministic HiGHS wrapper is pinned in requirements.
    https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.milp.html
    Solver optimality concerns this stated metadata objective, not model quality.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("Need three positive split ratios summing to one")
    names, sensors, frames, counts = _inventory(records)
    requirements, _ = _requirements(sensors, counts)
    eligible = [si for si in range(len(sensors)) if np.count_nonzero(counts[:, si, 2]) >= 3]
    features = [("global_frames", frames.sum(1), 4.)]
    features += [(f"{sensor}/frames", frames[:, si], 1. / len(sensors)) for si, sensor in enumerate(sensors)]
    features += [(f"global_class_{ci}", counts[:, :, ci].sum(1), .5) for ci in range(2)]
    features += [(f"{sensor}/class_{ci}", counts[:, si, ci], 1. / (2 * len(sensors)))
                 for si, sensor in enumerate(sensors) for ci in range(2)]
    features += [(f"{sensors[si]}/Car", counts[:, si, 2], .25 / len(eligible)) for si in eligible]
    ng, nf = len(names), len(features)
    nx, nv = 3 * ng, 3 * (ng + nf)
    cost, lower, upper = np.zeros(nv), np.zeros(nv), np.full(nv, np.inf)
    upper[:nx] = 1
    integer = np.zeros(nv, dtype=int)
    integer[:nx] = 1
    for gi, name in enumerate(names):
        for slot, split in enumerate(SPLITS):
            digest = hashlib.sha256(f"{seed}\n{name}\n{split}".encode()).digest()
            cost[gi * 3 + slot] = int.from_bytes(digest[:8], "big") / 2**64 * 1e-8 / nx
    matrix, lows, highs = [], [], []

    def constrain(row, low=-np.inf, high=np.inf):
        matrix.append(row)
        lows.append(low)
        highs.append(high)

    for gi in range(ng):
        row = np.zeros(nv)
        row[gi * 3:gi * 3 + 3] = 1
        constrain(row, 1, 1)
    for item, values in requirements:
        # Capping by the minimum preserves feasibility and avoids tiny coefficients.
        floor = item["minimum_points_per_split"]
        for slot in range(3):
            row = np.zeros(nv)
            row[slot:nx:3] = np.minimum(values, floor) / floor
            constrain(row, 1)
    for fi, (_, values, weight) in enumerate(features):
        fractions = values.astype(np.float64) / values.sum()
        for slot in range(3):
            slack = nx + fi * 3 + slot
            cost[slack] = weight
            row = np.zeros(nv)
            row[slot:nx:3] = fractions
            row[slack] = -1
            constrain(row, high=ratios[slot])
            row = np.zeros(nv)
            row[slot:nx:3] = -fractions
            row[slack] = -1
            constrain(row, high=-ratios[slot])
    result = milp(cost, integrality=integer, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(np.asarray(matrix), lows, highs),
                  options={"time_limit": 30., "node_limit": 100000, "mip_rel_gap": 1e-6})
    if not result.success or result.x is None:
        raise ValueError(f"Capture allocation failed (solver status {result.status}): {result.message}. "
                         "No fallback or model-score-based seed search was performed")
    binary = result.x[:nx].reshape(ng, 3)
    if not np.allclose(binary, np.rint(binary), atol=1e-5) or not np.allclose(binary.sum(1), 1):
        raise ValueError("Solver returned an invalid integral capture assignment")
    assignment = {name: SPLITS[int(binary[gi].argmax())] for gi, name in enumerate(names)}
    quality = split_quality([dict(r, split=assignment[r["group"]]) for r in records])
    quality.update(split_seed=int(seed), requested_ratios=ratios.tolist(),
                   solver={"status": int(result.status), "message": result.message,
                           "objective": float(result.fun), "mip_gap": float(result.mip_gap)},
                   balance_features=[name for name, _, _ in features])
    return assignment, quality
