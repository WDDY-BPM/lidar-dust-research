"""Short server-only throughput checks; no checkpoints or research metrics."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dust.data import IGNORE, read_manifest, write_json
from dust.dataset import DustDataset
from dust.evaluation import verify_records
from dust.training import EpochBatchSampler, initialize_worker
from models.pointnet2_dust_msg import get_model
from train_dust import device_for, seed_everything, source_version


def measure(args, manifest, batch_size, device):
    seed_everything(args.seed)
    dataset = DustDataset(args.data_root, manifest, seed=args.seed, samples_per_frame=4,
                          max_frames=args.max_frames, cache_frames=args.cache_frames)
    sampler = EpochBatchSampler(len(dataset), batch_size, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                        worker_init_fn=initialize_worker,
                        generator=torch.Generator().manual_seed(args.seed),
                        **({"prefetch_factor": 2} if args.workers else {}))
    model = get_model().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    weights = torch.tensor(manifest["class_weights"], dtype=torch.float32, device=device)
    iterator = iter(loader)
    epoch, samples = 0, 0
    healthy = torch.ones((), dtype=torch.bool, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        for step in range(args.warmup + args.steps):
            if step == args.warmup:
                sync()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = perf_counter()
            try:
                points, target = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                points, target = next(iterator)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(points.to(device, non_blocking=True).transpose(1, 2))
            numerator = F.nll_loss(prediction.reshape(-1, 2), target.flatten(),
                                   weight=weights, ignore_index=IGNORE, reduction="sum")
            denominator = torch.where(target != IGNORE, weights[target.clamp_min(0)], 0.).sum()
            loss = numerator / denominator.clamp_min(1e-12)
            healthy.logical_and_(torch.isfinite(loss) & (denominator > 0))
            loss.backward()
            optimizer.step()
            if step >= args.warmup:
                samples += len(points)
        sync()
        seconds = perf_counter() - started
        if not bool(healthy.item()):
            raise RuntimeError("Non-finite loss or all-ignore batch in benchmark")
        return {"batch_size": batch_size, "status": "ok", "steps": args.steps,
                "seconds": seconds, "seconds_per_batch": seconds / args.steps,
                "samples_per_second": samples / seconds,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None}
    finally:
        # Persistent workers must be retired before trying the next batch size,
        # including when the current size runs out of CUDA memory.
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


def main(args):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if (args.steps < 1 or args.warmup < 1 or args.max_frames < 1 or args.workers < 0 or
            args.cache_frames < 0 or any(b < 1 for b in args.batch_sizes)):
        raise ValueError("Invalid benchmark settings")
    if Path(args.output).exists():
        raise ValueError("Benchmark output exists; use a new JSON path")
    manifest = read_manifest(args.manifest)
    device = device_for(args.device)
    selected = DustDataset(args.data_root, manifest, max_frames=args.max_frames, seed=args.seed)
    verify_records(args.data_root, selected.frames)
    import json
    rows = []
    for batch_size in args.batch_sizes:
        print(f"Benchmark batch_size={batch_size}: warmup={args.warmup}, timed_steps={args.steps}", flush=True)
        try:
            row = measure(args, manifest, batch_size, device)
        except torch.cuda.OutOfMemoryError:
            row = {"batch_size": batch_size, "status": "cuda_out_of_memory"}
        rows.append(row)
        print(json.dumps(row), flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(args.output, {"is_benchmark": True, "validation_included": False,
                             "source": source_version(), "manifest_fingerprint": manifest["fingerprint"],
                             "device": str(device), "torch_version": str(torch.__version__),
                             "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                             "seed": args.seed, "workers": args.workers, "cache_frames": args.cache_frames,
                             "max_frames": args.max_frames, "results": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="outputs/throughput.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
