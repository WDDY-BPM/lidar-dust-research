"""Remote synthetic regression checks for graph inputs and narrow resume migration."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dust.evaluation import EvaluationInterrupted
from dust.inference import InferenceForward
from dust.resume import EVALUATION_UPDATE_PARENT, validate_resume_config
from models.fps_control import supplied_fps_starts
from models.pointnet2_utils import farthest_point_sample, index_points
from models.pointnet2_dust_msg import get_model
import test_dust_performance as performance
from test_dust_performance import StochasticTiny
from train_dust import train


class FourSamplingTiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 2, 1)
        for i, count in enumerate((8, 4, 2, 1), 1):
            setattr(self, f"sa{i}", SimpleNamespace(npoint=count))

    def forward(self, x):
        xyz = x[:, :3].transpose(1, 2)
        for i in range(1, 5):
            xyz = index_points(xyz, farthest_point_sample(xyz, getattr(self, f"sa{i}").npoint))
        offset = xyz.mean((1, 2))[:, None, None]
        logits = self.conv(x) + torch.cat((offset, -offset), dim=1)
        return F.log_softmax(logits, dim=1).transpose(1, 2), xyz


class ValidationTests(unittest.TestCase):
    def test_real_msg_explicit_starts_match_eager_forward_and_rng(self):
        model = get_model().eval()
        tensor = torch.randn(1, 4, 1024)
        with torch.inference_mode():
            torch.manual_seed(11)
            expected = model(tensor)[0]
            state = torch.get_rng_state().clone()
            torch.manual_seed(11)
            starts = [(count, torch.randint(0, count, (1,), dtype=torch.long))
                      for count in (1024, 1024, 256, 64)]
            with supplied_fps_starts(starts):
                actual = model(tensor)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_supplied_start_matches_eager_fps_and_preserves_cpu_rng(self):
        torch.manual_seed(3)
        xyz = torch.randn(2, 48, 3)
        torch.manual_seed(19)
        expected = farthest_point_sample(xyz, 24)
        state = torch.get_rng_state().clone()
        torch.manual_seed(19)
        start = torch.randint(0, 48, (2,), dtype=torch.long)
        with supplied_fps_starts([(48, start)]):
            actual = farthest_point_sample(xyz, 24)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_start_context_fails_closed_and_resets_after_exception(self):
        xyz = torch.zeros(2, 8, 3)
        for starts, message in (([], "more"), ([(9, torch.zeros(2, dtype=torch.long))], "shape")):
            with self.assertRaisesRegex(RuntimeError, message), supplied_fps_starts(starts):
                farthest_point_sample(xyz, 4)
        with self.assertRaisesRegex(RuntimeError, "fewer"), supplied_fps_starts([(8, torch.zeros(2))]):
            pass
        farthest_point_sample(xyz, 4)  # no leaked context after errors

    def test_graph_backend_rejects_cpu_and_eager_requires_eval(self):
        model = FourSamplingTiny()
        with self.assertRaisesRegex(ValueError, "requires a CUDA"):
            InferenceForward(model, torch.device("cpu"), 2, "cuda-graph")
        runner = InferenceForward(model, torch.device("cpu"), 2)
        with self.assertRaisesRegex(RuntimeError, "model.eval"):
            runner(torch.zeros(2, 4, 32))
        model.eval()
        runner(torch.zeros(2, 4, 32))
        self.assertEqual(runner.statistics()["eager_batches"], 1)

    def test_migration_only_accepts_clean_known_parent_and_identical_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            path.write_bytes(b"checkpoint hash input")
            old = {"source": {"commit": EVALUATION_UPDATE_PARENT, "tracked_files_dirty": False},
                   "batch_size": 32, "eval_batch_size": 4, "seed": 42}
            config = dict(old, source={"commit": "new", "tracked_files_dirty": False})
            checkpoint = {"config": old}
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_resume_config(checkpoint, config, path)
            lineage = validate_resume_config(checkpoint, config, path, True)
            self.assertEqual(lineage[0]["from_source"], old["source"])
            self.assertEqual(len(lineage[0]["checkpoint_sha256"]), 64)
            for wrong in (dict(config, batch_size=16), dict(config, eval_batch_size=32),
                          dict(config, source={"commit": "new", "tracked_files_dirty": True})):
                with self.assertRaisesRegex(ValueError, "identical"):
                    validate_resume_config(checkpoint, wrong, path, True)
            wrong_parent = copy.deepcopy(checkpoint)
            wrong_parent["config"]["source"]["commit"] = "unreviewed"
            with self.assertRaisesRegex(ValueError, "clean 4e7b7ae"):
                validate_resume_config(wrong_parent, config, path, True)
            self.assertEqual(validate_resume_config({"config": config, "resume_lineage": lineage},
                                                    config, path), lineage)

    def test_migration_resumes_validation_without_retraining_or_resetting_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, manifest, args = performance.PerformanceTests().setup_data(work)
            args.epochs = 1
            parent = {"commit": EVALUATION_UPDATE_PARENT, "tracked_files_dirty": False}
            updated = {"commit": "evaluation-update", "tracked_files_dirty": False}
            with patch("train_dust.get_model", StochasticTiny):
                with patch("train_dust.source_version", return_value=parent), patch(
                        "train_dust.evaluate", side_effect=EvaluationInterrupted("stop")):
                    train(args)
                before = torch.load(work / "run/last.pt", weights_only=False)
                args.resume = str(work / "run/last.pt")
                args.resume_evaluation_update = True
                def forbidden_step(optimizer, *a, **kw):
                    raise AssertionError("Must not retrain")
                with patch("train_dust.source_version", return_value=updated), patch.object(
                        torch.optim.Adam, "step", forbidden_step):
                    train(args)
            after = torch.load(work / "run/last.pt", weights_only=False)
            performance.PerformanceTests().assert_tree_equal(before["model_state"], after["model_state"])
            performance.PerformanceTests().assert_tree_equal(before["optimizer_state"], after["optimizer_state"])
            self.assertEqual((after["next_epoch"], after["next_batch"]), (1, 0))
            self.assertEqual(after["config"]["source"], updated)
            self.assertEqual(len(after["resume_lineage"]), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA parity runs on the designated GPU server")
    def test_cuda_graph_replay_matches_eager_rng_ragged_batches_and_weight_updates(self):
        device = torch.device("cuda:0")
        model = FourSamplingTiny().to(device).eval()
        runner = InferenceForward(model, device, 2, "cuda-graph")
        for index, batch in enumerate((2, 2, 1, 2)):
            tensor = torch.randn(batch, 4, 32, device=device)
            with torch.inference_mode():
                if index:
                    model.conv.weight.add_(.01)
                torch.manual_seed(27)
                gpu_before = torch.cuda.get_rng_state(device).clone()
                expected = model(tensor)[0].clone()
                state = torch.get_rng_state().clone()
                torch.manual_seed(27)
                actual = runner(tensor)[0].clone()
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertTrue(torch.equal(gpu_before, torch.cuda.get_rng_state(device)))
        self.assertEqual(runner.replays, 3)
        self.assertEqual(runner.eager_batches, 1)


if __name__ == "__main__":
    unittest.main()
