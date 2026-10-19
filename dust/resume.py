"""Explicit, narrow migrations for evaluation execution and validation cadence."""

from dust.data import file_sha256


EVALUATION_UPDATE_PARENT = "4e7b7ae65a0f36d8fbdc57b18a1af6b8737d65ce"
CUDA_GRAPH_PARENT = "27ee46bd97a0fdb1b798a7e0367b4785a296e755"


def validation_schedule(config, lineage):
    """Selection policy without seed-specific checkpoint hashes or source metadata."""
    changes = [{key: entry[key] for key in
                ("from_eval_every", "to_eval_every", "effective_from_epoch")}
               for entry in lineage if entry.get("kind") == "validation_schedule_update"]
    return {"initial_eval_every": changes[0]["from_eval_every"] if changes else config.get("eval_every"),
            "changes": changes}


def validate_resume_config(checkpoint, config, path, allow_evaluation_update=False,
                           allow_eval_schedule_update=False):
    previous = checkpoint["config"]
    lineage = list(checkpoint.get("resume_lineage", []))
    if previous == config:
        return lineage
    if allow_eval_schedule_update:
        old_source, new_source = previous["source"], config["source"]
        old_interval, new_interval = previous.get("eval_every"), config.get("eval_every")
        comparable = dict(previous, source=new_source, eval_every=new_interval)
        known_source = (old_source == new_source or old_source["commit"] in
                        (EVALUATION_UPDATE_PARENT, CUDA_GRAPH_PARENT))
        if (not known_source or old_source["tracked_files_dirty"] is not False
                or new_source["tracked_files_dirty"] is not False or not new_source["commit"]
                or type(old_interval) is not int or old_interval < 1
                or type(new_interval) is not int or new_interval < 1 or comparable != config):
            raise ValueError("Validation schedule update requires clean known source (4e7b7ae/27ee46b "
                             "or the same current commit), positive eval_every, and identical other settings")
        event = {"kind": "validation_schedule_update" if old_interval != new_interval else "evaluation_source_update",
                 "checkpoint_sha256": file_sha256(path), "from_source": old_source, "to_source": new_source,
                 "from_eval_every": old_interval, "to_eval_every": new_interval,
                 "effective_from_epoch": checkpoint["next_epoch"] + 1,
                 "next_batch": checkpoint["next_batch"],
                 "reason": "explicit validation cadence/source update; previous completed validation "
                           "candidates retained; model, optimizer, scheduler, RNG and cursor retained"}
        lineage.append(event)
        return lineage
    comparable = dict(previous, source=config["source"])
    if not allow_evaluation_update:
        raise ValueError("Resume config/source/manifest differs from checkpoint. For the audited "
                         "4e7b7ae execution update, use --resume-evaluation-update; to explicitly change "
                         "eval_every, use --resume-eval-schedule-update.")
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
