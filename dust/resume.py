"""Explicit, narrow source migration for the evaluation-only performance update."""

from dust.data import file_sha256


EVALUATION_UPDATE_PARENT = "4e7b7ae65a0f36d8fbdc57b18a1af6b8737d65ce"


def validate_resume_config(checkpoint, config, path, allow_evaluation_update=False):
    previous = checkpoint["config"]
    lineage = list(checkpoint.get("resume_lineage", []))
    if previous == config:
        return lineage
    comparable = dict(previous, source=config["source"])
    if not allow_evaluation_update:
        raise ValueError("Resume config/source/manifest differs from checkpoint. For the audited "
                         "4e7b7ae evaluation update only, use --resume-evaluation-update.")
    if (comparable != config or previous["source"] != {
            "commit": EVALUATION_UPDATE_PARENT, "tracked_files_dirty": False}
            or config["source"]["tracked_files_dirty"] is not False
            or not config["source"]["commit"]):
        raise ValueError("Evaluation update requires a clean 4e7b7ae checkpoint, clean current source, "
                         "and identical training/validation configuration")
    lineage.append({"checkpoint_sha256": file_sha256(path), "from_source": previous["source"],
                    "to_source": config["source"], "reason": "evaluation-only performance update; "
                    "optimizer, scheduler, RNG, epoch and committed batch cursor retained"})
    return lineage
