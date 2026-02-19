"""
CondSeg Training Script
========================
Full training loop for the CondSeg model with:
    - AdamW optimizer with MultiStepLR scheduler (per paper)
    - Mixed-precision training (AMP) for A100 efficiency
    - Gradient accumulation for effective larger batch sizes
    - Checkpoint saving (best + periodic)
    - TensorBoard + Weights & Biases (wandb) logging
    - Visual prediction logging (masks, ellipses) every N epochs
    - On-the-fly augmentation via dataset

Usage (single GPU):
    python train.py --data_root data --epochs 200 --batch_size 4 --img_size 1024

Usage (multi-GPU DDP):
    torchrun --nproc_per_node=2 train.py --data_root data --epochs 200 --batch_size 4

Logging:
    --logger wandb      → Weights & Biases (default)
    --logger tensorboard → TensorBoard only
    --logger both        → Both simultaneously
    --logger none        → No external logging

All arguments have sensible defaults matching the paper's recommendations.
"""

import os
import sys
import time
import math
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from condseg import CondSeg
from dataset import EyeSegmentationDataset


# ============================================================================
#  ARGUMENT PARSER
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CondSeg Training")

    # Data
    p.add_argument("--data_root", type=str, default="data",
                   help="Root directory with train/valid/test subdirs")
    p.add_argument("--img_size", type=int, default=1024,
                   help="Input image resolution (square)")

    # Training
    p.add_argument("--epochs", type=int, default=200,
                   help="Total training epochs")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size PER GPU")
    p.add_argument("--lr", type=float, default=4e-4,
                   help="Initial learning rate (paper: 0.0004)")
    p.add_argument("--weight_decay", type=float, default=1e-2,
                   help="AdamW weight decay")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers per GPU")

    # Model
    p.add_argument("--tau", type=float, default=800.0,
                   help="Ellp2Mask temperature")
    p.add_argument("--epsilon", type=float, default=0.01,
                   help="Minimum relative axis length for iris")
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Use pretrained EfficientNet-B3")
    p.add_argument("--no_pretrained", action="store_false", dest="pretrained",
                   help="Train from scratch")

    # AMP
    p.add_argument("--amp", action="store_true", default=True,
                   help="Enable mixed-precision training")
    p.add_argument("--no_amp", action="store_false", dest="amp",
                   help="Disable mixed-precision")

    # Gradient accumulation
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps (effective BS = batch_size * grad_accum * num_gpus)")

    # Checkpointing / Logging
    p.add_argument("--save_dir", type=str, default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--log_dir", type=str, default="runs",
                   help="TensorBoard log directory")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    # Logger selection
    p.add_argument("--logger", type=str, default="wandb",
                   choices=["wandb", "tensorboard", "both", "none"],
                   help="Logging backend: wandb, tensorboard, both, or none")
    p.add_argument("--wandb_project", type=str, default="CondSeg",
                   help="W&B project name")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="W&B run name (auto-generated if not set)")
    p.add_argument("--vis_every", type=int, default=5,
                   help="Log visual predictions every N epochs")

    return p.parse_args()


# ============================================================================
#  LOGGER WRAPPER (abstracts TensorBoard / W&B)
# ============================================================================

class TrainLogger:
    """
    Unified logging interface for TensorBoard and Weights & Biases.
    Only active on the main process (rank 0).
    """

    def __init__(self, args, rank):
        self.use_tb = False
        self.use_wandb = False
        self.rank = rank

        if rank != 0:
            return  # no logging on non-main processes

        # ---- TensorBoard ----
        if args.logger in ("tensorboard", "both"):
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=args.log_dir)
            self.use_tb = True
            print(f"  ✓ TensorBoard logging → {args.log_dir}")

        # ---- Weights & Biases ----
        if args.logger in ("wandb", "both"):
            try:
                import wandb
                self.wandb = wandb

                run_name = args.wandb_run_name or f"condseg_{args.img_size}_bs{args.batch_size}_lr{args.lr}"
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=vars(args),
                    resume="allow",
                )
                # Watch model gradients + parameters
                self.use_wandb = True
                print(f"  ✓ W&B logging → project={args.wandb_project}, run={run_name}")
            except ImportError:
                print("  ⚠ wandb not installed, falling back to tensorboard only")
                print("    Install with: pip install wandb")
                if not self.use_tb:
                    from torch.utils.tensorboard import SummaryWriter
                    self.tb_writer = SummaryWriter(log_dir=args.log_dir)
                    self.use_tb = True

    def watch_model(self, model):
        """W&B model watching (logs gradients + parameters)."""
        if self.use_wandb:
            unwrapped = model.module if isinstance(model, DDP) else model
            self.wandb.watch(unwrapped, log="gradients", log_freq=100)

    def log_scalars(self, metrics: dict, step: int):
        """Log scalar metrics to both backends."""
        if self.rank != 0:
            return
        if self.use_tb:
            for key, val in metrics.items():
                self.tb_writer.add_scalar(key, val, step)
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float, epoch_time: float):
        """Log epoch-level summary to both backends."""
        if self.rank != 0:
            return

        combined = {
            "epoch": epoch,
            "lr": lr,
            "epoch_time_s": epoch_time,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }

        if self.use_tb:
            for key, val in combined.items():
                self.tb_writer.add_scalar(key, val, epoch)
        if self.use_wandb:
            self.wandb.log(combined, step=epoch)

    @torch.no_grad()
    def log_predictions(self, model, dataloader, device, epoch, num_samples=4):
        """
        Log visual prediction samples with RANDOM selection each epoch.
        Shows: input image, GT masks, predicted masks, soft iris ellipse.
        """
        if self.rank != 0:
            return
        if not self.use_wandb and not self.use_tb:
            return

        import random

        model.eval()

        # ImageNet denormalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        # Pick random indices from the dataset
        dataset = dataloader.dataset
        total = len(dataset)
        sample_indices = random.sample(range(total), min(num_samples, total))

        for log_idx, data_idx in enumerate(sample_indices):
            image, gt_eye, gt_iris = dataset[data_idx]

            # Add batch dim and move to device
            image  = image.unsqueeze(0).to(device)
            gt_eye = gt_eye.unsqueeze(0).to(device)
            gt_iris = gt_iris.unsqueeze(0).to(device)

            with autocast('cuda', enabled=True):
                outputs = model(image) if not isinstance(model, DDP) else model.module(image)

            # Denormalize image
            img = image[0] * std + mean
            img = img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)

            # Extract masks
            pred_eye  = outputs["predicted_eye_region_mask"][0, 0].float().cpu().numpy()
            pred_iris = outputs["soft_iris_mask"][0, 0].float().cpu().numpy()
            pred_vis  = outputs["predicted_visible_iris"][0, 0].float().cpu().numpy()
            gt_eye_np = gt_eye[0, 0].cpu().numpy()
            gt_iris_np = gt_iris[0, 0].cpu().numpy()

            # Iris ellipse params
            params = outputs["iris_params"][0].detach().cpu().numpy()
            caption = (f"x0={params[0]:.0f} y0={params[1]:.0f} "
                       f"a={params[2]:.0f} b={params[3]:.0f} θ={params[4]:.2f}")

            if self.use_wandb:
                import wandb
                self.wandb.log({
                    f"predictions/sample_{log_idx}": [
                        wandb.Image(img, caption=f"Input (idx={data_idx})"),
                        wandb.Image(gt_eye_np, caption="GT Eye Region"),
                        wandb.Image(pred_eye, caption="Pred Eye Region"),
                        wandb.Image(gt_iris_np, caption="GT Visible Iris"),
                        wandb.Image(pred_vis, caption="Pred Visible Iris"),
                        wandb.Image(pred_iris, caption=f"Soft Iris | {caption}"),
                    ],
                }, step=epoch)

            if self.use_tb:
                self.tb_writer.add_image(f"pred/sample_{log_idx}/input",
                                          torch.from_numpy(img).permute(2, 0, 1), epoch)
                self.tb_writer.add_image(f"pred/sample_{log_idx}/gt_eye",
                                          torch.from_numpy(gt_eye_np).unsqueeze(0), epoch)
                self.tb_writer.add_image(f"pred/sample_{log_idx}/pred_eye",
                                          torch.from_numpy(pred_eye).unsqueeze(0), epoch)
                self.tb_writer.add_image(f"pred/sample_{log_idx}/gt_iris",
                                          torch.from_numpy(gt_iris_np).unsqueeze(0), epoch)
                self.tb_writer.add_image(f"pred/sample_{log_idx}/pred_vis_iris",
                                          torch.from_numpy(pred_vis).unsqueeze(0), epoch)
                self.tb_writer.add_image(f"pred/sample_{log_idx}/soft_iris",
                                          torch.from_numpy(pred_iris).unsqueeze(0), epoch)

        model.train()

    def log_best(self, val_loss: float, epoch: int):
        """Log best model event."""
        if self.rank != 0:
            return
        if self.use_wandb:
            self.wandb.run.summary["best_val_loss"] = val_loss
            self.wandb.run.summary["best_epoch"] = epoch

    def finish(self):
        """Close all loggers."""
        if self.rank != 0:
            return
        if self.use_tb:
            self.tb_writer.close()
        if self.use_wandb:
            self.wandb.finish()


# ============================================================================
#  DDP UTILITIES
# ============================================================================

def setup_ddp():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


# ============================================================================
#  LEARNING RATE SCHEDULER (per paper: MultiStepLR at 1/6, 1/3, 1/2, 5/6)
# ============================================================================

def get_milestones(total_epochs):
    """Compute LR reduction milestones at 1/6, 1/3, 1/2, 5/6 of total epochs."""
    fractions = [1/6, 1/3, 1/2, 5/6]
    return [int(f * total_epochs) for f in fractions]


# ============================================================================
#  TRAINING LOOP
# ============================================================================

def train_one_epoch(
    model, dataloader, optimizer, scaler, device, epoch,
    use_amp, grad_accum_steps, logger, global_step, rank,
):
    """Train for one epoch, return average loss and updated global_step."""
    model.train()
    total_loss_sum = 0.0
    eye_loss_sum = 0.0
    iris_loss_sum = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, (images, gt_eye, gt_iris) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        gt_eye = gt_eye.to(device, non_blocking=True)
        gt_iris = gt_iris.to(device, non_blocking=True)

        # Forward pass with optional AMP (model inference in fp16)
        with autocast('cuda', enabled=use_amp):
            outputs = model(images)

        # Loss computation OUTSIDE autocast (BCELoss requires float32)
        losses = model.module.compute_loss(outputs, gt_eye, gt_iris) if isinstance(model, DDP) \
                 else model.compute_loss(outputs, gt_eye, gt_iris)
        loss = losses["total_loss"] / grad_accum_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Accumulate losses for logging
        total_loss_sum += losses["total_loss"].item()
        eye_loss_sum   += losses["loss_eye"].item()
        iris_loss_sum  += losses["loss_iris"].item()
        num_batches += 1
        global_step += 1

        # Step-level scalar logging (every 20 steps)
        if is_main_process(rank) and batch_idx % 20 == 0:
            logger.log_scalars({
                "step/loss_total": losses["total_loss"].item(),
                "step/loss_eye": losses["loss_eye"].item(),
                "step/loss_iris": losses["loss_iris"].item(),
            }, step=global_step)

        # Print progress
        if is_main_process(rank) and batch_idx % 10 == 0:
            iris_params = outputs["iris_params"][0].detach().cpu()
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}]  "
                  f"loss={losses['total_loss'].item():.4f}  "
                  f"eye={losses['loss_eye'].item():.4f}  "
                  f"iris={losses['loss_iris'].item():.4f}  "
                  f"ellipse=[{iris_params[0]:.0f},{iris_params[1]:.0f},"
                  f"a={iris_params[2]:.0f},b={iris_params[3]:.0f},θ={iris_params[4]:.2f}]")

    avg_total = total_loss_sum / max(num_batches, 1)
    avg_eye   = eye_loss_sum / max(num_batches, 1)
    avg_iris  = iris_loss_sum / max(num_batches, 1)

    return avg_total, avg_eye, avg_iris, global_step


@torch.no_grad()
def validate(model, dataloader, device, use_amp):
    """Run validation, return average losses."""
    model.eval()
    total_loss_sum = 0.0
    eye_loss_sum = 0.0
    iris_loss_sum = 0.0
    num_batches = 0

    for images, gt_eye, gt_iris in dataloader:
        images = images.to(device, non_blocking=True)
        gt_eye = gt_eye.to(device, non_blocking=True)
        gt_iris = gt_iris.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(images)

        # Loss outside autocast for BCELoss compatibility
        losses = model.module.compute_loss(outputs, gt_eye, gt_iris) if isinstance(model, DDP) \
                 else model.compute_loss(outputs, gt_eye, gt_iris)

        total_loss_sum += losses["total_loss"].item()
        eye_loss_sum   += losses["loss_eye"].item()
        iris_loss_sum  += losses["loss_iris"].item()
        num_batches += 1

    avg_total = total_loss_sum / max(num_batches, 1)
    avg_eye   = eye_loss_sum / max(num_batches, 1)
    avg_iris  = iris_loss_sum / max(num_batches, 1)

    return avg_total, avg_eye, avg_iris


# ============================================================================
#  MAIN
# ============================================================================

def main():
    args = parse_args()

    # ---- DDP setup ----
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_ddp = world_size > 1

    if is_main_process(rank):
        print("=" * 70)
        print("CondSeg Training")
        print("=" * 70)
        print(f"  Device       : {device} (world_size={world_size})")
        print(f"  Data root    : {args.data_root}")
        print(f"  Image size   : {args.img_size}×{args.img_size}")
        print(f"  Batch size   : {args.batch_size} per GPU × {world_size} GPUs × {args.grad_accum} accum "
              f"= {args.batch_size * world_size * args.grad_accum} effective")
        print(f"  Epochs       : {args.epochs}")
        print(f"  LR           : {args.lr}")
        print(f"  AMP          : {args.amp}")
        print(f"  Logger       : {args.logger}")
        print(f"  τ            : {args.tau}")
        print(f"  ε            : {args.epsilon}")
        print()

    # ---- Logger (TensorBoard / W&B / both) ----
    logger = TrainLogger(args, rank)

    # ---- Datasets ----
    train_dataset = EyeSegmentationDataset(
        data_root=os.path.join(args.data_root, "train"),
        img_size=args.img_size,
        augment=True,
    )
    val_dataset = EyeSegmentationDataset(
        data_root=os.path.join(args.data_root, "valid"),
        img_size=args.img_size,
        augment=False,
    )

    # ---- DataLoaders ----
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)  if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    if is_main_process(rank):
        print(f"  Train : {len(train_dataset)} images, {len(train_loader)} batches/epoch")
        print(f"  Valid : {len(val_dataset)} images, {len(val_loader)} batches/epoch")
        print()

    # ---- Model ----
    model = CondSeg(
        img_size=args.img_size,
        tau=args.tau,
        epsilon=args.epsilon,
        pretrained=args.pretrained,
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if is_main_process(rank):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params : {total_params:,}")
        print()

    # Watch model in W&B (gradient histograms)
    logger.watch_model(model)

    # ---- Optimizer & Scheduler (per paper) ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    milestones = get_milestones(args.epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )
    if is_main_process(rank):
        print(f"  LR milestones: {milestones} (gamma=0.2)")
        print()

    # ---- AMP scaler ----
    scaler = GradScaler('cuda', enabled=args.amp)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume is not None and os.path.exists(args.resume):
        if is_main_process(rank):
            print(f"  Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model_state = ckpt.get("model_state_dict", ckpt)
        if isinstance(model, DDP):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main_process(rank):
            print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_val={best_val_loss:.6f}")
            print()

    # ---- Create save directory ----
    if is_main_process(rank):
        os.makedirs(args.save_dir, exist_ok=True)

    # ---- Training loop ----
    if is_main_process(rank):
        print("=" * 70)
        print("Starting training...")
        print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        if is_ddp:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss, train_eye, train_iris, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            args.amp, args.grad_accum, logger, global_step, rank,
        )

        # Validate
        val_loss, val_eye, val_iris = validate(model, val_loader, device, args.amp)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Epoch-level logging ----
        if is_main_process(rank):
            print(f"\n  Epoch {epoch}/{args.epochs-1}  ({epoch_time:.1f}s)  "
                  f"lr={current_lr:.6f}")
            print(f"    Train — total={train_loss:.4f} eye={train_eye:.4f} iris={train_iris:.4f}")
            print(f"    Valid — total={val_loss:.4f} eye={val_eye:.4f} iris={val_iris:.4f}")

            # Log epoch metrics
            logger.log_epoch(
                epoch=epoch,
                train_metrics={"loss": train_loss, "loss_eye": train_eye, "loss_iris": train_iris},
                val_metrics={"loss": val_loss, "loss_eye": val_eye, "loss_iris": val_iris},
                lr=current_lr,
                epoch_time=epoch_time,
            )

            # Log visual predictions periodically
            if (epoch + 1) % args.vis_every == 0 or epoch == 0:
                logger.log_predictions(model, val_loader, device, epoch, num_samples=4)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                }, os.path.join(args.save_dir, "best.pt"))
                print(f"    ★ New best model saved (val_loss={val_loss:.6f})")
                logger.log_best(val_loss, epoch)

            # Periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                }, os.path.join(args.save_dir, f"epoch_{epoch:04d}.pt"))
                print(f"    Checkpoint saved: epoch_{epoch:04d}.pt")

            print()

    # ---- Cleanup ----
    logger.finish()
    cleanup_ddp()

    if is_main_process(rank):
        print("=" * 70)
        print(f"Training complete! Best val loss: {best_val_loss:.6f}")
        print(f"Best model: {os.path.join(args.save_dir, 'best.pt')}")
        print("=" * 70)


if __name__ == "__main__":
    main()
