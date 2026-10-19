"""Synthetic regression tests for operator equivalence and committed-step resume."""

import argparse
import copy
import json
import os
import signal
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import benchmark_dust
from dust.dataset import DustDataset
from dust.evaluation import EvaluationInterrupted
from dust.training import EpochBatchSampler
from models.pointnet2_utils import farthest_point_sample, query_ball_point, square_distance
from prepare_dust import prepare
from train_dust import train


class StochasticTiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 2, 1)
        self.dropout = torch.nn.Dropout(.25)

    def forward(self, x):
        return F.log_softmax(self.dropout(self.conv(x)), dim=1).transpose(1, 2), None


def original_fps(xyz, npoint):
    """Frozen pre-optimization implementation, for compatibility regression."""
    batch, count, _ = xyz.shape
    centroids = torch.zeros(batch, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(batch, count).to(xyz.device) * 1e10
    farthest = torch.randint(0, count, (batch,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(batch, dtype=torch.long).to(xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)

    def setup_data(self, work):
        root, prepared = work / "raw", work / "prepared"
        for sensor, base in (("m1", 0), ("ouster", 10)):
            folder = root / sensor
            folder.mkdir(parents=True)
            for sequence in range(base, base + 3):
                frame = folder / f"sequence_{sequence:04d}_{sensor}_0001_100_000.bin"
                np.array([[0, 0, 0, 2], [1, 1, 1, 3 + sequence], [2, 2, 0, 7],
                          [np.nan, 0, 0, 0]], dtype="<f4").tofile(frame)
                np.array([1, 9, 25, 0], dtype=np.uint8).tofile(frame.with_suffix(".label"))
        manifest = prepare(argparse.Namespace(data_root=str(root), output=str(prepared), group_map=None,
                           seed=42, train_ratio=.7, val_ratio=.15, npoint=1024,
                           block_size=10., stride=5., coord_scale=10.))
        args = argparse.Namespace(data_root=str(root), manifest=str(prepared / "manifest.json"),
                                  output=str(work / "run"), resume=None, device="cpu", epochs=3,
                                  batch_size=2, workers=0, samples_per_frame=3, seed=42,
                                  lr=.001, weight_decay=.0001, threshold=.5, eval_every=2,
                                  eval_batch_size=2, max_train_frames=2, max_val_frames=2,
                                  no_augment=False, save_every_steps=2, log_every=2,
                                  stop_after_steps=None, init_checkpoint=None)
        return root, manifest, args

    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_fps_matches_original_on_random_duplicate_and_tied_points(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))
        for device in devices:
            torch.manual_seed(4)
            xyz = torch.randn(2, 48, 3).to(device)
            xyz[:, 5:10] = xyz[:, :1]
            for value in (xyz, torch.zeros_like(xyz)):
                torch.manual_seed(19)
                expected = original_fps(value, 24)
                state_after = torch.get_rng_state().clone()
                torch.manual_seed(19)
                actual = farthest_point_sample(value, 24)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.get_rng_state(), state_after))

    def test_ball_query_preserves_input_index_order_boundary_and_padding(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))
        for device in devices:
            xyz = torch.tensor([[[0., 0, 0], [1., 0, 0], [.1, 0, 0], [0, 0, 0],
                                 [5., 0, 0], [-1., 0, 0], [10., 0, 0], [.5, 0, 0]]], device=device)
            centers = xyz[:, [0, 4]]
            for count in (3, 6, 8):
                indices = torch.arange(8, device=device).view(1, 1, 8).repeat(1, 2, 1)
                indices[square_distance(centers, xyz) > 1.] = 8
                expected = indices.sort(dim=-1).values[:, :, :count]
                first = expected[:, :, :1].expand_as(expected)
                expected = torch.where(expected == 8, first, expected)
                actual = query_ball_point(1., count, xyz, centers)
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(query_ball_point(1., 3, xyz, centers)[0, 0].tolist(), [0, 1, 2])

    def test_sampler_committed_cursor_is_independent_of_prefetch(self):
        sampler = EpochBatchSampler(23, 4, 42)
        sampler.set_epoch(3)
        complete = list(sampler)
        iterator = iter(sampler)
        for _ in range(5):
            next(iterator)  # worker queue advances farther than committed steps
        sampler.set_epoch(3, 2)
        self.assertEqual(list(sampler), complete[2:])
        self.assertEqual(sorted(i for batch in complete for epoch, i in batch), list(range(23)))
        sampler.set_epoch(4)
        self.assertNotEqual([i for b in sampler for _, i in b], [i for b in complete for _, i in b])

    def test_persistent_workers_receive_new_epoch_and_resume_same_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, args = self.setup_data(Path(directory))
            dataset = DustDataset(root, manifest, samples_per_frame=1)
            sampler = EpochBatchSampler(len(dataset), 2, 42)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, persistent_workers=True)
            try:
                for epoch in (0, 1):
                    sampler.set_epoch(epoch)
                    actual = list(loader)
                    for (points, labels), batch in zip(actual, sampler):
                        expected = [dataset[index] for index in batch]
                        np.testing.assert_array_equal(points.numpy(), np.stack([x for x, _ in expected]))
                        np.testing.assert_array_equal(labels.numpy(), np.stack([y for _, y in expected]))
            finally:
                loader._iterator._shutdown_workers()

    def test_mid_epoch_resume_matches_uninterrupted_with_dropout_and_workers(self):
        for workers in (0, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                root, manifest, args = self.setup_data(work)
                args.workers = workers
                args.stop_after_steps = 2
                with patch("train_dust.get_model", StochasticTiny):
                    train(args)
                    partial = torch.load(work / "run" / "last.pt", weights_only=False)
                    self.assertEqual((partial["next_epoch"], partial["next_batch"]), (0, 2))
                    self.assertFalse((work / "run" / "best.pt").exists())
                    args.resume = str(work / "run" / "last.pt")
                    args.stop_after_steps = None
                    train(args)
                    args.resume = None
                    args.output = str(work / "full")
                    train(args)
                resumed = torch.load(work / "run" / "last.pt", weights_only=False)
                full = torch.load(work / "full" / "last.pt", weights_only=False)
                for key in ("model_state", "optimizer_state", "scheduler_state", "epoch_statistics"):
                    self.assert_tree_equal(resumed[key], full[key])
                self.assertTrue(torch.equal(resumed["rng_state"]["torch"], full["rng_state"]["torch"]))
                self.assertEqual((resumed["next_epoch"], resumed["next_batch"]), (3, 0))
                histories = [[json.loads(line) for line in (work / name / "history.jsonl").read_text().splitlines()]
                             for name in ("run", "full")]
                for a, b in zip(*histories):
                    for key in ("epoch", "lr", "train_loss", "val"):
                        self.assertEqual(a[key], b[key])

    def test_validation_interrupt_retries_without_an_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, manifest, args = self.setup_data(work)
            args.epochs = 1
            with patch("train_dust.get_model", StochasticTiny):
                with patch("train_dust.evaluate", side_effect=EvaluationInterrupted("stop")):
                    train(args)
                partial = torch.load(work / "run" / "last.pt", weights_only=False)
                self.assertEqual((partial["next_epoch"], partial["next_batch"]), (0, 3))
                args.resume = str(work / "run" / "last.pt")
                def forbidden_step(optimizer, *a, **kw):
                    raise AssertionError("Must not retrain")
                with patch.object(torch.optim.Adam, "step", forbidden_step), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    train(args)
                self.assertFalse(any("lr_scheduler.step()" in str(w.message) for w in caught))
                args.resume, args.output = None, str(work / "full")
                train(args)
            resumed = torch.load(work / "run" / "last.pt", weights_only=False)
            full = torch.load(work / "full" / "last.pt", weights_only=False)
            self.assert_tree_equal(resumed["model_state"], full["model_state"])
            self.assert_tree_equal(resumed["scheduler_state"], full["scheduler_state"])

    def test_sigint_saves_after_the_committed_step(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, manifest, args = self.setup_data(work)
            original_step = torch.optim.Adam.step

            def interrupt_after_step(optimizer, *a, **kw):
                result = original_step(optimizer, *a, **kw)
                os.kill(os.getpid(), signal.SIGINT)
                return result

            with patch("train_dust.get_model", StochasticTiny):
                with patch.object(torch.optim.Adam, "step", interrupt_after_step):
                    train(args)
                checkpoint = torch.load(work / "run" / "last.pt", weights_only=False)
                self.assertEqual((checkpoint["next_epoch"], checkpoint["next_batch"]), (0, 1))
                args.resume = str(work / "run" / "last.pt")
                train(args)
            self.assertEqual(torch.load(work / "run" / "last.pt", weights_only=False)["next_epoch"], 3)

    def test_history_and_best_recover_from_interrupted_epoch_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, manifest, args = self.setup_data(work)
            args.epochs = 1
            with patch("train_dust.get_model", StochasticTiny):
                train(args)
                (work / "run" / "history.jsonl").write_text('{"epoch":')
                (work / "run" / "best.pt").unlink()
                args.resume, args.epochs = str(work / "run" / "last.pt"), 2
                train(args)
            rows = [json.loads(line) for line in (work / "run" / "history.jsonl").read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in rows], [1, 2])
            self.assertTrue((work / "run" / "best.pt").is_file())

    def test_old_weights_are_explicit_initialization_not_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, manifest, args = self.setup_data(work)
            old_folder = work / "old"
            old_folder.mkdir()
            old_path = old_folder / "last.pt"
            torch.save({"format_version": 1, "epoch": 49, "model_state": StochasticTiny().state_dict(),
                        "config": {"manifest_fingerprint": manifest["fingerprint"],
                                   "source": {"commit": "old", "tracked_files_dirty": False}}}, old_path)
            args.output, args.resume = str(old_folder), str(old_path)
            with patch("train_dust.get_model", StochasticTiny):
                with self.assertRaisesRegex(ValueError, "Old checkpoint"):
                    train(args)
                args.output, args.resume, args.init_checkpoint = str(work / "new"), None, str(old_path)
                args.epochs = 1
                train(args)
            state = torch.load(work / "new" / "last.pt", weights_only=False)
            self.assertEqual(state["epoch"], 0)
            self.assertEqual(state["config"]["initialization"]["mode"], "weights_only_fresh_optimizer")
            self.assertEqual(state["config"]["initialization"]["source"]["commit"], "old")

    def test_benchmark_cli_reports_throughput_without_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, manifest, train_args = self.setup_data(work)
            args = argparse.Namespace(data_root=str(root), manifest=train_args.manifest,
                                      output=str(work / "throughput.json"), device="cpu", batch_sizes=[2, 4],
                                      workers=0, cache_frames=2, max_frames=2, warmup=1, steps=2, seed=42)
            with patch("benchmark_dust.get_model", StochasticTiny):
                benchmark_dust.main(args)
            result = json.loads(Path(args.output).read_text())
            self.assertTrue(result["is_benchmark"])
            self.assertFalse(result["validation_included"])
            self.assertEqual([r["status"] for r in result["results"]], ["ok", "ok"])
            self.assertTrue(all(r["samples_per_second"] > 0 for r in result["results"]))
            self.assertFalse(list(work.rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
