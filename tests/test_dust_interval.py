"""Remote synthetic checks for explicit validation-interval continuation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dust.evaluation import EvaluationInterrupted
from dust.resume import (CUDA_GRAPH_PARENT, EVALUATION_UPDATE_PARENT,
                         validate_resume_config, validation_schedule)
from summarize_dust import summarize_runs
import test_dust_performance as performance
from test_dust_performance import StochasticTiny
import train_dust


class IntervalTests(unittest.TestCase):
    def test_only_interval_and_known_source_can_change_with_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            path.write_bytes(b"checkpoint hash input")
            for parent in (CUDA_GRAPH_PARENT, EVALUATION_UPDATE_PARENT):
                old = {"source": {"commit": parent, "tracked_files_dirty": False},
                       "eval_every": 5, "batch_size": 32, "eval_batch_size": 4,
                       "seed": 42, "lr": .001, "threshold": .5, "manifest_fingerprint": "same"}
                config = dict(old, eval_every=10, source={"commit": "interval-update", "tracked_files_dirty": False})
                checkpoint = {"config": old, "next_epoch": 9, "next_batch": 651}
                with self.assertRaisesRegex(ValueError, "differs"):
                    validate_resume_config(checkpoint, config, path)
                with self.assertRaises(ValueError):
                    validate_resume_config(checkpoint, config, path, allow_evaluation_update=True)
                lineage = validate_resume_config(checkpoint, config, path, allow_eval_schedule_update=True)
                self.assertEqual(validation_schedule(config, lineage), {
                    "initial_eval_every": 5, "changes": [{"from_eval_every": 5,
                    "to_eval_every": 10, "effective_from_epoch": 10}]})
                self.assertEqual(lineage[0]["next_batch"], 651)
                self.assertEqual(len(lineage[0]["checkpoint_sha256"]), 64)
                self.assertEqual(old["eval_every"], 5)
                for change in ({"batch_size": 16}, {"eval_batch_size": 32}, {"seed": 43},
                               {"lr": .01}, {"threshold": .6}, {"manifest_fingerprint": "changed"},
                               {"eval_every": 0}, {"eval_every": True}):
                    with self.assertRaisesRegex(ValueError, "identical other"):
                        validate_resume_config(checkpoint, dict(config, **change), path,
                                               allow_eval_schedule_update=True)
                for source in ({"commit": parent, "tracked_files_dirty": True},
                               {"commit": "unknown", "tracked_files_dirty": False}):
                    invalid = dict(checkpoint, config=dict(old, source=source))
                    with self.assertRaisesRegex(ValueError, "clean known"):
                        validate_resume_config(invalid, config, path, allow_eval_schedule_update=True)

    def test_mid_batch_change_keeps_initial_state_and_matches_full_training(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, manifest, args = performance.PerformanceTests().setup_data(work)
            args.eval_every, args.stop_after_steps = 5, 2
            parent = {"commit": CUDA_GRAPH_PARENT, "tracked_files_dirty": False}
            updated = {"commit": "interval-update", "tracked_files_dirty": False}
            with patch("train_dust.get_model", StochasticTiny):
                with patch("train_dust.source_version", return_value=parent):
                    train_dust.train(args)
                before = torch.load(work / "run/last.pt", weights_only=False)
                args.resume, args.stop_after_steps = str(work / "run/last.pt"), None
                args.eval_every, args.resume_eval_schedule_update = 10, True
                initial = []
                save = train_dust.atomic_checkpoint

                def audit_save(path, state):
                    if not initial:
                        for key in ("model_state", "optimizer_state", "scheduler_state"):
                            performance.PerformanceTests().assert_tree_equal(before[key], state[key])
                        self.assertTrue(torch.equal(before["rng_state"]["torch"], state["rng_state"]["torch"]))
                        self.assertTrue(torch.equal(before["rng_state"]["loader"], state["rng_state"]["loader"]))
                        self.assertEqual((state["next_epoch"], state["next_batch"]), (0, 2))
                        self.assertEqual(state["epoch_statistics"], before["epoch_statistics"])
                        initial.append(True)
                    save(path, state)

                with patch("train_dust.source_version", return_value=updated):
                    with patch("train_dust.atomic_checkpoint", audit_save):
                        train_dust.train(args)
                    args.resume, args.output = None, str(work / "full")
                    args.resume_eval_schedule_update = False
                    train_dust.train(args)
            resumed = torch.load(work / "run/last.pt", weights_only=False)
            full = torch.load(work / "full/last.pt", weights_only=False)
            self.assertTrue(initial)
            for key in ("model_state", "optimizer_state", "scheduler_state"):
                performance.PerformanceTests().assert_tree_equal(resumed[key], full[key])
            self.assertTrue(torch.equal(resumed["rng_state"]["torch"], full["rng_state"]["torch"]))
            self.assertEqual((resumed["next_epoch"], resumed["next_batch"]), (3, 0))
            rows = [json.loads(line) for line in (work / "run/history.jsonl").read_text().splitlines()]
            self.assertEqual([row["val"] is not None for row in rows], [False, False, True])
            self.assertEqual(resumed["validation_schedule"]["initial_eval_every"], 5)

    def test_pending_validation_uses_new_schedule_retains_best_and_can_resume_again(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, manifest, args = performance.PerformanceTests().setup_data(work)
            args.epochs, args.eval_every = 4, 1
            parent = {"commit": CUDA_GRAPH_PARENT, "tracked_files_dirty": False}
            updated = {"commit": "interval-update", "tracked_files_dirty": False}
            evaluate = train_dust.evaluate

            def stop_at_third(*a, **kw):
                if kw["output"].name == "epoch_0003":
                    raise EvaluationInterrupted("pause during old scheduled validation")
                return evaluate(*a, **kw)

            with patch("train_dust.get_model", StochasticTiny):
                with patch("train_dust.source_version", return_value=parent), patch(
                        "train_dust.evaluate", stop_at_third):
                    train_dust.train(args)
                before = torch.load(work / "run/last.pt", weights_only=False)
                old_best = (work / "run/best.pt").read_bytes()
                self.assertEqual((before["next_epoch"], before["next_batch"]), (2, 3))
                args.resume = str(work / "run/last.pt")
                args.eval_every, args.resume_eval_schedule_update, args.stop_after_steps = 2, True, 1
                with patch("train_dust.source_version", return_value=updated):
                    with patch("train_dust.evaluate", side_effect=AssertionError("Epoch 3 must skip validation")):
                        train_dust.train(args)
                    after = torch.load(work / "run/last.pt", weights_only=False)
                    self.assertEqual((after["next_epoch"], after["next_batch"]), (3, 1))
                    self.assertEqual(after["scheduler_state"]["last_epoch"],
                                     before["scheduler_state"]["last_epoch"] + 1)
                    self.assertEqual((work / "run/best.pt").read_bytes(), old_best)
                    rows = [json.loads(line) for line in (work / "run/history.jsonl").read_text().splitlines()]
                    self.assertEqual([row["epoch"] for row in rows], [1, 2, 3])
                    self.assertIsNone(rows[-1]["val"])
                    args.resume_eval_schedule_update, args.stop_after_steps = False, None
                    with patch("train_dust.evaluate", wraps=evaluate) as final_validation:
                        train_dust.train(args)
                    self.assertEqual(final_validation.call_count, 1)
                    self.assertEqual(final_validation.call_args.kwargs["output"].name, "epoch_0004")
            complete = torch.load(work / "run/last.pt", weights_only=False)
            self.assertEqual((complete["next_epoch"], complete["next_batch"]), (4, 0))
            self.assertEqual(len(complete["resume_lineage"]), 1)

    def test_summary_rejects_different_selection_histories_with_same_final_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (42, 43, 44):
                report = {"split": "test", "is_smoke_run": False, "training_seed": seed,
                          "manifest_fingerprint": "same", "threshold": .5, "frames": 20,
                          "checkpoint_source": {"commit": "same"}, "evaluation_seed": 42,
                          "evaluation_batch_size": 4, "training_budget_epochs": 100,
                          "training_config": {"seed": seed, "eval_every": 10},
                          "overall": {"f1": .8, "precision": .7, "recall": .9,
                                      "dust_iou": .6, "false_deletion_rate": .01}}
                path = Path(directory) / f"seed{seed}.json"
                path.write_text(json.dumps(report))
                paths.append(path)
            summarize_runs(paths)
            report["resume_lineage"] = [{"kind": "validation_schedule_update", "from_eval_every": 5,
                                         "to_eval_every": 10, "effective_from_epoch": 10}]
            paths[-1].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "validation schedule histories"):
                summarize_runs(paths)


if __name__ == "__main__":
    unittest.main()
