"""Synthetic software checks; no dataset downloads or research experiments."""

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from dust.data import (IGNORE, LABEL_PROTOCOL, binary_labels, evaluation_blocks,
                       finite_frame, load_frame, read_manifest)
from dust.dataset import DustDataset
from dust.evaluation import predict_frame, evaluate
from dust.metrics import confusion, summarize
from models.pointnet2_dust_msg import get_model
from models.pointnet2_sem_seg_msg import get_model as OriginalMSG
from models.pointnet2_utils import square_distance
from prepare_dust import assign_splits, prepare
from train_dust import train
from evaluate_dust import main as evaluate_main
from summarize_dust import summarize_runs


class ConstantDust(torch.nn.Module):
    def forward(self, x):
        probability = torch.tensor([.75, .25], device=x.device).log()
        return probability.expand(x.shape[0], x.shape[2], 2), None


class TinyNetwork(torch.nn.Module):
    """Fast stand-in to check loop/checkpoint wiring independently of MSG."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 2, 1)

    def forward(self, x):
        return F.log_softmax(self.conv(x), dim=1).transpose(1, 2), None


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mapping_and_shared_nonfinite_mask(self):
        labels = np.arange(29, dtype=np.uint8)
        target = binary_labels(labels)
        self.assertEqual(target[9], 1)
        self.assertTrue(np.all(target[[0, 10, 11, 25]] == IGNORE))
        self.assertEqual(np.sum(target == 0), 24)
        with self.assertRaises(ValueError):
            binary_labels(np.array([29], dtype=np.uint8))
        points = np.arange(116, dtype=np.float32).reshape(29, 4)
        points[5, 1] = np.nan
        points[9, 3] = np.inf
        filtered, filtered_target, index = finite_frame(points, labels)
        np.testing.assert_array_equal(index, np.delete(np.arange(29), [5, 9]))
        np.testing.assert_array_equal(filtered, points[index])
        np.testing.assert_array_equal(filtered_target, target[index])

    def test_known_confusion_and_undefined_metrics(self):
        truth = np.array([1, 1, 0, 0, IGNORE])
        score = np.array([.9, .1, .8, .2, np.nan])
        counts = confusion(truth, score)
        np.testing.assert_array_equal(counts, [1, 1, 1, 1])
        metrics = summarize(counts)
        self.assertEqual(metrics["f1"], .5)
        self.assertEqual(metrics["false_deletion_rate"], .5)
        self.assertIsNone(summarize([5, 0, 0, 0])["f1"])
        with self.assertRaises(ValueError):
            confusion(np.array([0]), np.array([np.nan]))

    def test_coverage_including_dense_sparse_negative_and_boundary_points(self):
        rng = np.random.default_rng(1)
        points = rng.uniform(-11, 11, (2500, 4)).astype(np.float32)
        points[:4, :2] = [[-10, -10], [0, 0], [5, 5], [10, 10]]
        points[-1, :2] = [10000, -10000]  # one isolated tile, no empty-grid explosion
        geometry = {"npoint": 1024, "block_size": 10., "stride": 5., "coord_scale": 10.}
        counts = np.zeros(len(points), dtype=int)
        for index, real_count, center in evaluation_blocks(points, geometry, rng):
            self.assertEqual(len(index), 1024)
            self.assertEqual(len(np.unique(index[:real_count])), real_count)
            self.assertTrue(np.isfinite(center).all())
            np.add.at(counts, index[:real_count], 1)
        np.testing.assert_array_equal(counts, np.full(len(points), 4))
        manifest = {"geometry": geometry, "intensity": {"test": {"mean": 0., "std": 1.}}}
        record = {"path": "test/frame.bin", "sensor": "test"}
        torch.manual_seed(123)
        state = torch.get_rng_state().clone()
        result, votes = predict_frame(ConstantDust(), points, record, manifest,
                                      torch.device("cpu"), batch_size=3)
        np.testing.assert_allclose(result, .25)
        np.testing.assert_array_equal(votes, counts)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def _make_data(self, root):
        for sensor, base in (("m1", 0), ("ouster", 10)):
            folder = root / sensor
            folder.mkdir(parents=True)
            for sequence in range(base, base + 3):
                frame = folder / f"sequence_{sequence:04d}_{sensor}_0001_{1609459200 + sequence * 86400}_000.bin"
                points = np.array([[0, 0, 0, 1 + sequence], [1, 1, 1, 4 + sequence],
                                   [2, 2, 0, 7 + sequence], [np.nan, 0, 0, 0]], dtype="<f4")
                points.tofile(frame)
                np.array([1, 9, 25, 0], dtype=np.uint8).tofile(frame.with_suffix(".label"))

    def _prepare(self, root, output):
        args = argparse.Namespace(data_root=str(root), output=str(output), group_map=None,
                                  seed=42, train_ratio=.7, val_ratio=.15, npoint=1024,
                                  block_size=10., stride=5., coord_scale=10.)
        return prepare(args)

    def test_prepare_train_only_stats_hashes_dataset_and_full_frame_export(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, artifacts = work / "raw", work / "prepared"
            self._make_data(root)
            manifest = self._prepare(root, artifacts)
            loaded = read_manifest(artifacts / "manifest.json")
            self.assertEqual(loaded["fingerprint"], manifest["fingerprint"])
            self.assertEqual(loaded["labels"], LABEL_PROTOCOL)
            for sensor in ("m1", "ouster"):
                training = next(r for r in loaded["frames"] if r["split"] == "train" and r["sensor"] == sensor)
                points, _ = load_frame(root, training, verify_hash=True)
                logged = np.log1p(points[:3, 3].astype(np.float64))
                self.assertAlmostEqual(loaded["intensity"][sensor]["mean"], logged.mean())
                self.assertAlmostEqual(loaded["intensity"][sensor]["std"], logged.std())
            dataset = DustDataset(root, loaded, samples_per_frame=1)
            x, y = dataset[0]
            self.assertEqual(x.shape, (1024, 4))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(np.any(y != IGNORE))
            x_again, y_again = dataset[0]
            np.testing.assert_array_equal(x, x_again)
            np.testing.assert_array_equal(y, y_again)
            records = [r for r in loaded["frames"] if r["split"] == "test"]
            report = evaluate(ConstantDust(), root, loaded, records, torch.device("cpu"),
                              output=work / "results", export_per_sensor=1)
            self.assertEqual(report["overall"]["tn"], 2)
            self.assertEqual(report["overall"]["fn"], 2)
            self.assertEqual(report["overall"]["n_evaluated"], 4)
            arrays = list((work / "results").rglob("*_prediction.npz"))
            self.assertEqual(len(arrays), 2)
            with np.load(arrays[0]) as prediction:
                np.testing.assert_array_equal(prediction["finite_index"], [0, 1, 2])
                self.assertTrue(np.isnan(prediction["dust_probability"][3]))
            filtered = list((work / "results").rglob("*_filtered.bin"))[0]
            self.assertEqual(filtered.stat().st_size, 3 * 16)  # ignored GT retained via prediction
            # Same-size mutation is caught by hash verification.
            frame = root / loaded["frames"][0]["path"]
            frame.write_bytes(b"\0" * frame.stat().st_size)
            with self.assertRaises(ValueError):
                load_frame(root, loaded["frames"][0], verify_hash=True)
            broken = json.loads((artifacts / "manifest.json").read_text())
            broken["geometry"]["coord_scale"] = 20.
            (artifacts / "tampered.json").write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                read_manifest(artifacts / "tampered.json")

    def test_same_capture_stays_together(self):
        records = [{"sensor": sensor, "group": f"capture_{capture}"}
                   for capture in range(10) for sensor in ("a", "b")]
        splits = assign_splits(records, 42, (.7, .15, .15))
        self.assertEqual(splits, assign_splits(records, 42, (.7, .15, .15)))
        self.assertEqual(set(splits.values()), {"train", "val", "test"})
        self.assertEqual(len(splits), 10)

    def test_train_resume_and_evaluate_cli_wiring(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            root, artifacts = work / "raw", work / "prepared"
            self._make_data(root)
            self._prepare(root, artifacts)
            args = argparse.Namespace(data_root=str(root), manifest=str(artifacts / "manifest.json"),
                                      output=str(work / "resumed"), resume=None, device="cpu",
                                      epochs=1, batch_size=2, workers=0, samples_per_frame=1,
                                      seed=42, lr=.001, weight_decay=.0001, threshold=.5,
                                      eval_every=5, eval_batch_size=2, max_train_frames=2,
                                      max_val_frames=2, no_augment=False)
            with patch("train_dust.get_model", TinyNetwork):
                train(args)
                args.epochs = 2
                args.resume = str(work / "resumed" / "last.pt")
                train(args)
                args.resume = None
                args.output = str(work / "uninterrupted")
                train(args)
            resumed = torch.load(work / "resumed" / "last.pt", weights_only=False)
            uninterrupted = torch.load(work / "uninterrupted" / "last.pt", weights_only=False)
            self.assertEqual(resumed["epoch"], 1)
            self.assertEqual(resumed["scheduler_state"], uninterrupted["scheduler_state"])
            for key in resumed["model_state"]:
                torch.testing.assert_close(resumed["model_state"][key],
                                           uninterrupted["model_state"][key], rtol=0, atol=0)
            evaluation_args = argparse.Namespace(data_root=str(root), manifest=str(artifacts / "manifest.json"),
                                                 checkpoint=str(work / "resumed" / "best.pt"),
                                                 output=str(work / "evaluation"), split="test", device="cpu",
                                                 max_frames=None, visualize_per_sensor=0, allow_smoke=False)
            with patch("evaluate_dust.get_model", TinyNetwork):
                with self.assertRaises(ValueError):
                    evaluate_main(evaluation_args)
                evaluation_args.allow_smoke = True
                evaluate_main(evaluation_args)
            metrics = json.loads((work / "evaluation" / "metrics.json").read_text())
            self.assertEqual(metrics["split"], "test")
            self.assertTrue(metrics["is_smoke_run"])
            self.assertEqual(metrics["overall"]["n_evaluated"], 4)

    def test_squared_distance_is_nonnegative_at_duplicate_points(self):
        xyz = torch.randn(2, 32, 3) * 100
        distance = square_distance(xyz, xyz)
        self.assertTrue(torch.all(distance >= 0))
        reference = ((xyz[:, :, None] - xyz[:, None, :])**2).sum(-1)
        torch.testing.assert_close(distance, reference, rtol=1e-5, atol=.02)

    def test_summary_rejects_different_training_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            template = {"split": "test", "is_smoke_run": False, "manifest_fingerprint": "same",
                        "threshold": .5, "frames": 20, "checkpoint_source": {"commit": "same"},
                        "evaluation_seed": 42, "evaluation_batch_size": 4, "training_budget_epochs": 100,
                        "overall": {"f1": .8, "precision": .7, "recall": .9,
                                    "dust_iou": .6, "false_deletion_rate": .01}}
            for seed in (42, 43, 44):
                report = copy.deepcopy(template)
                report["training_seed"] = seed
                report["training_config"] = {"seed": seed, "lr": .001}
                path = Path(directory) / f"seed{seed}.json"
                path.write_text(json.dumps(report))
                paths.append(path)
            result = summarize_runs(paths)
            self.assertAlmostEqual(result["metrics"]["f1"]["mean"], .8)
            self.assertAlmostEqual(result["metrics"]["f1"]["sample_std"], 0.)
            changed = json.loads(paths[0].read_text())
            changed["training_config"]["lr"] = .01
            paths[0].write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                summarize_runs(paths)

    def test_real_msg_shape_and_backward_and_s3dis_default(self):
        self.assertEqual(OriginalMSG(13).sa1.conv_blocks[0][0].in_channels, 12)
        model = get_model()
        self.assertEqual(model.sa1.conv_blocks[0][0].in_channels, 7)
        torch.manual_seed(7)
        log_prob, _ = model(torch.randn(2, 4, 1024))
        self.assertEqual(tuple(log_prob.shape), (2, 1024, 2))
        torch.testing.assert_close(log_prob.exp().sum(-1), torch.ones(2, 1024))
        target = torch.randint(0, 2, (2, 1024))
        target[:, :10] = IGNORE
        loss = F.nll_loss(log_prob.reshape(-1, 2), target.flatten(), ignore_index=IGNORE)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.conv2.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
