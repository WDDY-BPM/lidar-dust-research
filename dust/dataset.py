"""Lazy per-frame random spatial blocks with epoch/index deterministic sampling."""

from collections import OrderedDict

import numpy as np
from torch.utils.data import Dataset

from dust.data import IGNORE, features, finite_frame, load_frame


class DustDataset(Dataset):
    def __init__(self, root, manifest, samples_per_frame=4, seed=42, cache_frames=2,
                 max_frames=None, augment=True):
        self.root, self.manifest = root, manifest
        self.frames = [r for r in manifest["frames"] if r["split"] == "train"
                       and sum(r["class_counts"]) > 0]
        if max_frames is not None:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.permutation(len(self.frames))[:max_frames])
            self.frames = [self.frames[i] for i in indices]
        if not self.frames or samples_per_frame < 1 or cache_frames < 0:
            raise ValueError("Need supervised training frames and positive samples_per_frame")
        self.samples_per_frame, self.seed = samples_per_frame, seed
        self.cache_frames, self.cache = cache_frames, OrderedDict()
        self.epoch, self.augment = 0, augment

    def __len__(self):
        return len(self.frames) * self.samples_per_frame

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _frame(self, record):
        key = record["path"]
        if key not in self.cache:
            points, labels = load_frame(self.root, record)
            points, target, _ = finite_frame(points, labels)
            result = points, target, np.flatnonzero(target != IGNORE)
            if self.cache_frames == 0:
                return result
            self.cache[key] = result
            if len(self.cache) > self.cache_frames:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, index):
        record = self.frames[index // self.samples_per_frame]
        points, target, supervised = self._frame(record)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        geometry = self.manifest["geometry"]
        anchor = int(rng.choice(supervised))
        half = geometry["block_size"] / 2
        center = points[anchor, :3].copy()
        inside = np.flatnonzero(np.all(np.abs(points[:, :2] - center[:2]) <= half, axis=1))
        center[2] = points[inside, 2].mean(dtype=np.float64)
        # Always retain one supervised point, preventing all-ignore loss batches.
        others = inside[inside != anchor]
        npoint = geometry["npoint"]
        if len(others) >= npoint - 1:
            chosen = np.concatenate([[anchor], rng.choice(others, npoint - 1, replace=False)])
        else:
            chosen = np.concatenate([[anchor], others,
                                     rng.choice(inside, npoint - 1 - len(others), replace=True)])
        rng.shuffle(chosen)
        inputs = features(points[chosen], center, self.manifest["intensity"][record["sensor"]], geometry)
        if self.augment:
            angle = rng.uniform(0, 2 * np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]], dtype=np.float32)
            inputs[:, :2] = inputs[:, :2] @ rotation.T
        return inputs, target[chosen].copy()
