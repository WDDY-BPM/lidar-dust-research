"""Train the LiDARDustX PointNet++ MSG baseline in the designated environment."""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dust.data import IGNORE, read_manifest, write_json
from dust.dataset import DustDataset
from dust.evaluation import evaluate, select_frames, verify_records
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


def train(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or
            args.samples_per_frame < 1 or args.eval_every < 1 or args.eval_batch_size < 1 or
            not 0 < args.threshold < 1 or args.lr <= 0 or args.weight_decay < 0 or
            args.seed < 0 or any(v is not None and v < 1 for v in
                                 (args.max_train_frames, args.max_val_frames))):
        raise ValueError("Invalid training configuration")
    manifest = read_manifest(args.manifest)
    output = Path(args.output)
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise ValueError("Run output already exists; use --resume last.pt or a new directory")
    if args.resume is not None and Path(args.resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume must use last.pt in the same output directory")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = device_for(args.device)
    dataset = DustDataset(args.data_root, manifest, args.samples_per_frame, args.seed,
                          max_frames=args.max_train_frames, augment=not args.no_augment)
    validation = select_frames(manifest, "val", args.max_val_frames, seed=42)
    verify_records(args.data_root, dataset.frames + validation)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", drop_last=False, generator=generator,
                        persistent_workers=False)
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
              "evaluation_seed": 42, "is_smoke_run": smoke, "source": source_version()}
    start, best = 0, -1.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume config/source/manifest differs from checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng(checkpoint["rng_state"], generator)
        start, best = checkpoint["epoch"] + 1, checkpoint["best_val_f1"]
        if args.epochs <= start:
            raise ValueError("--epochs must exceed completed epochs")
    write_json(output / "config.json", {**config, "epochs_requested": args.epochs,
               "workers": args.workers, "device": str(device), "geometry": manifest["geometry"],
               "class_weights": manifest["class_weights"], "torch_version": str(torch.__version__),
               "numpy_version": np.__version__, "cuda_runtime": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    # Store the exact portable protocol for later evaluation, independently of
    # changes to a manifest file at the original path.
    write_json(output / "manifest.json", manifest)
    for epoch in range(start, args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        loss_sum, weight_sum = 0., 0.
        learning_rate = optimizer.param_groups[0]["lr"]
        for step, (points, target) in enumerate(loader):
            inputs = points.to(device, dtype=torch.float32, non_blocking=True).transpose(1, 2)
            target = target.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            log_prob, _ = model(inputs)
            numerator = F.nll_loss(log_prob.reshape(-1, 2), target.reshape(-1),
                                   weight=weights, ignore_index=IGNORE, reduction="sum")
            denominator = weights[target[target != IGNORE]].sum()
            if denominator.item() <= 0:
                raise RuntimeError("All-ignore training batch")
            loss = numerator / denominator
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += numerator.item()
            weight_sum += denominator.item()
            if step % 25 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} step {step + 1}/{len(loader)} loss={loss.item():.6f}", flush=True)
        scheduler.step()
        validation_report, is_best = None, False
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            validation_report = evaluate(model, args.data_root, manifest, validation, device,
                                         threshold=args.threshold, batch_size=args.eval_batch_size,
                                         seed=42, output=output / "validation" / f"epoch_{epoch + 1:04d}")
            score = validation_report["overall"]["f1"]
            score = -1. if score is None else score
            if score > best or not (output / "best.pt").exists():
                best, is_best = score, True
        state = {"format_version": 1, "epoch": epoch, "best_val_f1": best, "config": config,
                 "training_budget_epochs": args.epochs,
                 "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(), "rng_state": rng_state(generator),
                 "manifest": manifest}
        # Replace completed checkpoints atomically, keeping the previous one if
        # the process is interrupted while writing.
        torch.save(state, output / "last.pt.tmp")
        (output / "last.pt.tmp").replace(output / "last.pt")
        if is_best:
            torch.save(state, output / "best.pt.tmp")
            (output / "best.pt.tmp").replace(output / "best.pt")
        row = {"epoch": epoch + 1, "lr": learning_rate, "train_loss": loss_sum / weight_sum,
               "val": validation_report["overall"] if validation_report else None}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
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
    parser.add_argument("--max-train-frames", type=int)
    parser.add_argument("--max-val-frames", type=int)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", help="Resume this run's last.pt at an epoch boundary")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
