"""Conservative acquisition-day groups and non-destructive duplicate quarantine."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path


CAPTURE_POLICY = {
    "version": 2,
    "name": "capture_day_disjoint_v2",
    "clock_seconds_range": [1500000000, 1900000000],
    "clock_zone": "UTC",
    "groups": "connected sequences linked by UTC day, point hash, or complete user capture map",
    "duplicate_policy": "same points/labels: canonical path; conflicting labels: quarantine all copies",
    "unreliable_clock_policy": "quarantine unless a complete user capture map is supplied",
    "claim": "acquisition-day separation; physical site/scene independence is not established",
}


def capture_day(path):
    try:
        seconds = int(Path(path).stem.rsplit("_", 2)[-2])
    except (ValueError, IndexError):
        return None
    low, high = CAPTURE_POLICY["clock_seconds_range"]
    if not low <= seconds < high:
        return None
    return datetime.fromtimestamp(seconds, timezone.utc).date().isoformat()


class _Components:
    def __init__(self, names):
        self.parent = {name: name for name in names}

    def find(self, name):
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def join(self, names):
        roots = sorted({self.find(name) for name in names})
        for root in roots[1:]:
            self.parent[root] = roots[0]


def repair_capture_records(records, mapping=None):
    """Return fresh eligible records and an audit; never modify/delete raw files.

    Link complete sequences before removing duplicate/conflicting frames, so
    neighboring frames of duplicated captures cannot migrate across splits.
    No prediction, outcome metric, old split or model seed affects these rules.
    """
    sequences = {r["sequence"] for r in records}
    if mapping is not None and set(mapping) != sequences:
        raise ValueError("Group map must describe exactly the dataset's sequences")
    components = _Components(sequences)
    days, hashes, manual = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in records:
        day = capture_day(row["path"])
        if day is not None:
            days[day].append(row["sequence"])
        hashes[row["point_sha256"]].append(row)
        if mapping is not None:
            manual[mapping[row["sequence"]]].append(row["sequence"])
    for names in list(days.values()) + list(manual.values()):
        components.join(names)
    for rows in hashes.values():
        components.join([r["sequence"] for r in rows])
    members = defaultdict(list)
    for sequence in sorted(sequences):
        members[components.find(sequence)].append(sequence)
    group_for = {}
    for names in members.values():
        name = "capture_" + hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
        group_for.update({sequence: name for sequence in names})

    reasons, aliases, duplicates = defaultdict(set), {}, []
    for row in records:
        if capture_day(row["path"]) is None and mapping is None:
            reasons[row["path"]].add("unreliable_clock_without_capture_map")
    for digest, rows in sorted(hashes.items()):
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r["path"])
        conflict = len({r["label_sha256"] for r in rows}) > 1
        canonical = None
        if conflict:
            for row in rows:
                reasons[row["path"]].add("identical_points_conflicting_labels")
        else:
            eligible = [r for r in rows if not reasons[r["path"]]]
            canonical = eligible[0]["path"] if eligible else rows[0]["path"]
            for row in rows:
                if row["path"] != canonical:
                    reasons[row["path"]].add("duplicate_identical_points_and_labels")
                    aliases[row["path"]] = canonical
        duplicates.append({"point_sha256": digest,
                           "action": "quarantine_conflict" if conflict else "keep_canonical",
                           "canonical_path": canonical,
                           "members": [{k: r[k] for k in ("path", "sequence", "sensor", "label_sha256")}
                                       for r in rows]})

    retained, excluded = [], []
    for original in sorted(records, key=lambda r: r["path"]):
        row = dict(original)
        row.pop("split", None)
        row["capture_day"] = capture_day(row["path"])
        row["group"] = group_for[row["sequence"]]
        if reasons[row["path"]]:
            excluded.append({"path": row["path"], "sensor": row["sensor"],
                             "sequence": row["sequence"], "group": row["group"],
                             "reasons": sorted(reasons[row["path"]]),
                             "canonical_path": aliases.get(row["path"])})
        else:
            retained.append(row)
    active_paths = {r["path"] for r in retained}
    groups = []
    for names in sorted(members.values(), key=lambda x: x[0]):
        rows = [r for r in records if r["sequence"] in names]
        groups.append({"group": group_for[names[0]], "sequences": names,
                       "sensors": sorted({r["sensor"] for r in rows}),
                       "utc_days": sorted({capture_day(r["path"]) for r in rows
                                           if capture_day(r["path"]) is not None}),
                       "input_frames": len(rows), "retained_frames": sum(r["path"] in active_paths for r in rows)})
    report = {"policy": CAPTURE_POLICY, "input_frames": len(records), "retained_frames": len(retained),
              "excluded_frames": len(excluded), "duplicate_point_groups": len(duplicates),
              "conflicting_label_groups": sum(d["action"] == "quarantine_conflict" for d in duplicates),
              "exclusion_reason_counts": dict(Counter(reason for r in excluded for reason in r["reasons"])),
              "groups": groups, "duplicates": duplicates, "excluded": excluded}
    if not retained:
        raise ValueError("No eligible frames after capture/duplicate audit; review clock metadata or capture map")
    assert_clean_capture_records(retained, mapping is not None)
    return retained, report


def assert_clean_capture_records(records, allow_unreliable_clock=False):
    hashes, day_groups = set(), {}
    for row in records:
        if row["point_sha256"] in hashes:
            raise ValueError("Duplicate point cloud remains in clean manifest")
        hashes.add(row["point_sha256"])
        day = capture_day(row["path"])
        if row.get("capture_day") != day:
            raise ValueError("Capture day does not match frame timestamp")
        if day is None and not allow_unreliable_clock:
            raise ValueError("Unreliable clock requires an explicit capture map")
        if day is not None:
            if day in day_groups and day_groups[day] != row["group"]:
                raise ValueError("One acquisition day occurs in multiple capture groups")
            day_groups[day] = row["group"]


def write_capture_reports(output, report):
    from dust.data import write_json
    output = Path(output)
    write_json(output / "capture_audit.json", report)
    with (output / "capture_groups.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", "group"))
        for group in report["groups"]:
            writer.writerows((sequence, group["group"]) for sequence in group["sequences"])
    with (output / "excluded_frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sensor", "sequence", "group", "reasons", "canonical_path"))
        for row in report["excluded"]:
            writer.writerow((row["path"], row["sensor"], row["sequence"], row["group"],
                             ";".join(row["reasons"]), row["canonical_path"] or ""))
