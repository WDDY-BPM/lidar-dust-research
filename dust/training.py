"""Training order and cooperative stopping, independent of worker prefetch."""

import math
import signal

import torch


def initialize_worker(worker_id):
    # Only the parent handles cooperative stopping. Inheriting its SIGTERM
    # handler would prevent DataLoader from retiring a stalled worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


class EpochBatchSampler:
    """Recreate an epoch's permutation and skip only committed optimizer steps."""

    def __init__(self, size, batch_size, seed):
        if size < 1 or batch_size < 1:
            raise ValueError("Need a nonempty dataset and positive batch size")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.total_batches = math.ceil(size / batch_size)
        self.set_epoch(0)

    def set_epoch(self, epoch, start_batch=0):
        if epoch < 0 or not 0 <= start_batch <= self.total_batches:
            raise ValueError("Invalid epoch/batch cursor")
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for start in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield [(self.epoch, i) for i in order[start:start + self.batch_size]]

    def __len__(self):
        return self.total_batches - self.start_batch


class StopRequest:
    """SIGINT/SIGTERM finish the current optimizer step before saving state."""

    def __init__(self):
        self.requested = False
        self.previous = {}

    def _request(self, signum, frame):
        self.requested = True

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, self._request)
        return self

    def __exit__(self, *exc):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)

    def __call__(self):
        return self.requested
