"""Train the LiDARDustX PointNet++ MSG baseline in the designated environment."""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dust.data import IGNORE, file_sha256, read_manifest, write_json
from dust.dataset import DustDataset
from dust.evaluation import EvaluationInterrupted, evaluate, select_frames, verify_records
from dust.training import EpochBatchSampler, StopRequest, initialize_worker
from dust.resume import validate_resume_config
from models.pointnet2_dust_msg import get_model


def source_version():
    root = Path(__file__).resolve().parent
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                            cwd=root, stderr=subprocess.DEVNULL).strip())
        return {"commit": sha, "tracked_files_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_files_dirty": None}


def device_for(name):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use a CPU or CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu only for server-side small checks")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def reconcile_history(output, completed_epochs, row):
    """Checkpoint is authoritative if a crash separates it from the JSONL write."""
    path = output / "history.jsonl"
    rows = {}
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    break  # interrupted append, recover the committed row below
                raise
            if value["epoch"] <= completed_epochs:
                rows[value["epoch"]] = value
    if row is not None:
        rows[row["epoch"]] = row
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(rows[k], allow_nan=False) + "\n" for k in sorted(rows)),
                         encoding="utf-8")
    temporary.replace(path)


def train(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or
            args.samples_per_frame < 1 or args.eval_every < 1 or args.eval_batch_size < 1 or
            not 0 < args.threshold < 1 or args.lr <= 0 or args.weight_decay < 0 or
            args.seed < 0 or any(v is not None and v < 1 for v in
                                 (args.max_train_frames, args.max_val_frames))):
        raise ValueError("Invalid training configuration")
    save_every = getattr(args, "save_every_steps", 100)
    log_every = getattr(args, "log_every", 25)
    cache_frames = getattr(args, "cache_frames", 16)
    prefetch = getattr(args, "prefetch_factor", 2)
    stop_after = getattr(args, "stop_after_steps", None)
    init_path = getattr(args, "init_checkpoint", None)
    eval_backend = getattr(args, "eval_backend", "eager")
    evaluation_update = getattr(args, "resume_evaluation_update", False)
    if (save_every < 1 or log_every < 1 or cache_frames < 0 or prefetch < 1 or
            (stop_after is not None and stop_after < 1) or (args.resume and init_path)):
        raise ValueError("Invalid checkpoint/loader settings or conflicting resume/init options")
    if eval_backend not in ("eager", "cuda-graph") or (evaluation_update and not args.resume):
        raise ValueError("Invalid evaluation backend or evaluation update without --resume")
    manifest = read_manifest(args.manifest)
    output = Path(args.output)
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise ValueError("Run output already exists; use --resume last.pt or a new directory")
    if args.resume is not None and Path(args.resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume must use last.pt in the same output directory")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = device_for(args.device)
    if eval_backend == "cuda-graph" and device.type != "cuda":
        raise ValueError("cuda-graph evaluation requires CUDA")
    dataset = DustDataset(args.data_root, manifest, args.samples_per_frame, args.seed,
                          max_frames=args.max_train_frames, augment=not args.no_augment,
                          cache_frames=cache_frames)
    validation = select_frames(manifest, "val", args.max_val_frames, seed=42)
    verify_records(args.data_root, dataset.frames + validation)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = EpochBatchSampler(len(dataset), args.batch_size, args.seed)
    loader_options = {"prefetch_factor": prefetch} if args.workers else {}
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda", generator=generator,
                        persistent_workers=args.workers > 0, worker_init_fn=initialize_worker,
                        **loader_options)
    model = get_model().to(device)
    weights = torch.tensor(manifest["class_weights"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(.7**(epoch // 10), .01))
    smoke = args.max_train_frames is not None or args.max_val_frames is not None
    config = {"manifest_fingerprint": manifest["fingerprint"], "model": "pointnet2_dust_msg",
              "seed": args.seed, "batch_size": args.batch_size, "samples_per_frame": args.samples_per_frame,
              "lr": args.lr, "weight_decay": args.weight_decay, "threshold": args.threshold,
              "augmentation": "z_rotation" if not args.no_augment else "none",
              "max_train_frames": args.max_train_frames, "max_val_frames": args.max_val_frames,
              "eval_every": args.eval_every, "eval_batch_size": args.eval_batch_size,
              "evaluation_seed": 42, "is_smoke_run": smoke, "source": source_version(),
              "training_order": "epoch_permutation_v1", "initialization": None}
    start, start_batch, best = 0, 0, -1.
    saved_stats, saved_seconds, history_row = [0., 0.], 0., None
    resume_lineage = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["format_version"] != 2:
            raise ValueError("Old checkpoint: use the original code to resume, or --init-checkpoint "
                             "in a NEW output directory to initialize weights with a fresh optimizer")
        config["initialization"] = checkpoint["config"].get("initialization")
        resume_lineage = validate_resume_config(checkpoint, config, args.resume, evaluation_update)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng(checkpoint["rng_state"], generator)
        start, start_batch = checkpoint["next_epoch"], checkpoint["next_batch"]
        if start_batch:
            # LambdaLR tracks this non-serialized flag on its optimizer. A
            # validation-only retry has already committed optimizer steps in a
            # previous process, so its first scheduler step is not premature.
            optimizer._opt_called = True
        best = checkpoint["best_val_f1"]
        saved_stats, saved_seconds = checkpoint["epoch_statistics"], checkpoint["epoch_train_seconds"]
        history_row = checkpoint.get("history_row")
        sampler.set_epoch(start, start_batch)
        reconcile_history(output, start, history_row)
        if checkpoint.get("best_pending", False):
            atomic_checkpoint(output / "best.pt", checkpoint)
        if args.epochs <= start:
            raise ValueError("--epochs must exceed completed epochs")
        print(f"Resuming epoch {start + 1}, next batch {start_batch + 1}/{sampler.total_batches}", flush=True)
    elif init_path:
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        if checkpoint["config"]["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Initialization checkpoint and manifest differ")
        model.load_state_dict(checkpoint["model_state"])
        config["initialization"] = {"sha256": file_sha256(init_path),
                                    "source": checkpoint["config"]["source"],
                                    "epoch": checkpoint["epoch"] + 1,
                                    "mode": "weights_only_fresh_optimizer"}
        print("Initialized model weights only; optimizer, scheduler, epoch and best F1 start fresh.", flush=True)
    write_json(output / "config.json", {**config, "epochs_requested": args.epochs,
               "eval_backend": eval_backend, "resume_lineage": resume_lineage,
               "workers": args.workers, "cache_frames": cache_frames, "prefetch_factor": prefetch,
               "save_every_steps": save_every, "log_every": log_every,
               "device": str(device), "geometry": manifest["geometry"],
               "class_weights": manifest["class_weights"], "torch_version": str(torch.__version__),
               "numpy_version": np.__version__, "cuda_runtime": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    # Store the exact portable protocol for later evaluation, independently of
    # changes to a manifest file at the original path.
    write_json(output / "manifest.json", manifest)
    print(f"Training frames={len(dataset.frames)}, samples/epoch={len(dataset)}, "
          f"batches/epoch={sampler.total_batches}, batch_size={args.batch_size}, "
          f"checkpoint every {save_every} steps", flush=True)
    invocation_steps = 0
    with StopRequest() as stop:
        for epoch in range(start, args.epochs):
            cursor = start_batch if epoch == start else 0
            sampler.set_epoch(epoch, cursor)
            dataset.set_epoch(epoch)
            model.train()
            statistics = torch.tensor(saved_stats if epoch == start else [0., 0.],
                                      dtype=torch.float64, device=device)
            healthy = torch.ones((), dtype=torch.bool, device=device)
            seconds_before = saved_seconds if epoch == start else 0.
            epoch_clock, window_clock = perf_counter(), perf_counter()
            window_batch, window_samples = cursor, 0
            learning_rate = optimizer.param_groups[0]["lr"]

            def snapshot(next_epoch, next_batch, row=None, best_pending=False):
                if not bool(healthy.item()):
                    raise RuntimeError("Non-finite loss or all-ignore batch; last good checkpoint preserved")
                values = statistics.cpu().tolist()
                if next_batch and (values[1] <= 0 or not np.isfinite(values).all()):
                    raise RuntimeError("Invalid training statistics; last good checkpoint preserved")
                return {"format_version": 2, "epoch": epoch, "next_epoch": next_epoch,
                        "next_batch": next_batch, "best_val_f1": best, "config": config,
                        "training_budget_epochs": args.epochs, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                        "rng_state": rng_state(generator), "manifest": manifest,
                        "eval_backend": eval_backend, "resume_lineage": resume_lineage,
                        "epoch_statistics": values if next_batch else [0., 0.],
                        "epoch_train_seconds": seconds_before + perf_counter() - epoch_clock if next_batch else 0.,
                        "history_row": history_row if row is None else row, "best_pending": best_pending}

            # There is now a recoverable initial state even before the first
            # periodic checkpoint. Prefetch never advances the committed cursor.
            atomic_checkpoint(output / "last.pt", snapshot(epoch, cursor))
            if stop():
                print("Stop requested; checkpoint saved before the next batch.", flush=True)
                return
            for points, target in loader:
                if stop():
                    atomic_checkpoint(output / "last.pt", snapshot(epoch, cursor))
                    print("Stop requested; checkpoint saved before the next optimizer step.", flush=True)
                    return
                inputs = points.to(device, dtype=torch.float32, non_blocking=True).transpose(1, 2)
                target = target.to(device, dtype=torch.long, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                log_prob, _ = model(inputs)
                numerator = F.nll_loss(log_prob.reshape(-1, 2), target.reshape(-1),
                                       weight=weights, ignore_index=IGNORE, reduction="sum")
                # Fixed-size indexing, including ignored points: no CUDA nonzero
                # or CPU scalar read in the ordinary per-batch loss path.
                denominator = torch.where(target != IGNORE, weights[target.clamp_min(0)], 0.).sum()
                loss = numerator / denominator.clamp_min(1e-12)
                healthy.logical_and_(torch.isfinite(loss) & (denominator > 0))
                loss.backward()
                optimizer.step()
                statistics.add_(torch.stack((numerator.detach(), denominator.detach())).to(torch.float64))
                cursor += 1
                invocation_steps += 1
                window_samples += points.shape[0]
                stopping = stop() or (stop_after is not None and invocation_steps >= stop_after)
                periodic = cursor % save_every == 0
                if cursor == 1 or cursor % log_every == 0 or periodic or stopping or cursor == sampler.total_batches:
                    state = snapshot(epoch, cursor)
                    elapsed = max(perf_counter() - window_clock, 1e-9)
                    batch_seconds = elapsed / max(cursor - window_batch, 1)
                    avg_loss = state["epoch_statistics"][0] / state["epoch_statistics"][1]
                    if periodic or stopping:
                        # Save before logging: a closed output pipe must not
                        # discard the just-committed batch on a stop request.
                        atomic_checkpoint(output / "last.pt", state)
                    print(f"Epoch {epoch + 1}/{args.epochs} step {cursor}/{sampler.total_batches} "
                          f"loss={avg_loss:.6f} s/batch={batch_seconds:.3f} "
                          f"samples/s={window_samples / elapsed:.2f} "
                          f"train_ETA_min={(sampler.total_batches - cursor) * batch_seconds / 60:.1f}", flush=True)
                    window_clock, window_batch, window_samples = perf_counter(), cursor, 0
                if stopping:
                    print(f"Stopped safely. Resume from {output / 'last.pt'} (next batch {cursor + 1}).", flush=True)
                    return

            state = snapshot(epoch, cursor)
            train_seconds = state["epoch_train_seconds"]
            atomic_checkpoint(output / "last.pt", state)  # validation can be retried without retraining
            validation_report, is_best = None, False
            validation_clock = perf_counter()
            if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                print(f"Full validation: {len(validation)} frames (separate from training ETA)", flush=True)
                try:
                    validation_report = evaluate(model, args.data_root, manifest, validation, device,
                                                 threshold=args.threshold, batch_size=args.eval_batch_size,
                                                 seed=42, output=output / "validation" / f"epoch_{epoch + 1:04d}",
                                                 should_stop=stop, backend=eval_backend)
                except EvaluationInterrupted:
                    atomic_checkpoint(output / "last.pt", snapshot(epoch, cursor))
                    print("Stopped during validation. Resume retries validation, without repeating training.", flush=True)
                    return
                validation_report.update(evaluation_source=config["source"], checkpoint_epoch=epoch + 1)
                write_json(output / "validation" / f"epoch_{epoch + 1:04d}" / "metrics.json", validation_report)
                score = validation_report["overall"]["f1"]
                score = -1. if score is None else score
                if score > best or not (output / "best.pt").exists():
                    best, is_best = score, True
            validation_seconds = perf_counter() - validation_clock if validation_report else 0.
            scheduler.step()
            values = statistics.cpu().tolist()
            row = {"epoch": epoch + 1, "lr": learning_rate, "train_loss": values[0] / values[1],
                   "val": validation_report["overall"] if validation_report else None,
                   "train_seconds": train_seconds, "validation_seconds": validation_seconds}
            state = snapshot(epoch + 1, 0, row=row, best_pending=is_best)
            atomic_checkpoint(output / "last.pt", state)
            if is_best:
                atomic_checkpoint(output / "best.pt", state)
            history_row = row
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if stop():
                print("Stop requested; completed epoch checkpoint saved.", flush=True)
                return
    print(f"Finished. Best validation checkpoint: {output / 'best.pt'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-frame", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-backend", choices=("eager", "cuda-graph"), default="eager",
                        help="Execution backend only; keeps the same FPS draws, blocks and batch size")
    parser.add_argument("--resume-evaluation-update", action="store_true",
                        help="Explicit source migration from clean 4e7b7ae; retains all training state")
    parser.add_argument("--max-train-frames", type=int)
    parser.add_argument("--max-val-frames", type=int)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", help="Resume this run's last.pt from the next committed batch")
    parser.add_argument("--init-checkpoint", help="Weights only, fresh optimizer/epoch, NEW output directory")
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--cache-frames", type=int, default=16, help="Cached frames per worker")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--stop-after-steps", type=int, help="Save and exit after this many steps in this invocation")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
