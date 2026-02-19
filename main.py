"""
CondSeg Demo / Main Script
============================
Instantiates the full CondSeg model, creates dummy input tensors,
runs a training forward pass, prints all intermediate shapes,
and calls loss.backward() to verify full differentiability.

This script serves as both a smoke test and a reference for how
to integrate the CondSeg pipeline into a real training loop.

Usage:
    python main.py
"""

import torch
import time
import sys

# Add current dir to path for imports
sys.path.insert(0, ".")

from condseg import CondSeg
from dataset import create_dummy_gt_masks


def main():
    # ---- Configuration ----
    # NOTE: B=2 at 1024×1024 requires ~20GB VRAM (A100-class).
    # For verification on consumer GPUs (e.g. RTX 3050/4050), use B=1.
    BATCH_SIZE = 1
    IMG_SIZE   = 1024
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    # Clear any cached GPU memory
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("=" * 70)
    print("CondSeg — Full Pipeline Verification")
    print("=" * 70)
    print(f"  Device     : {DEVICE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Image size : {IMG_SIZE}×{IMG_SIZE}")
    print()

    # ---- Step 1: Instantiate model ----
    print("[1/5] Instantiating CondSeg model...")
    model = CondSeg(
        img_size=IMG_SIZE,
        epsilon=0.01,
        pretrained=True,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print()

    # ---- Step 2: Create dummy inputs ----
    print("[2/5] Creating dummy input tensors...")
    dummy_images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    gt_eye_region, gt_visible_iris = create_dummy_gt_masks(
        BATCH_SIZE, IMG_SIZE, IMG_SIZE, device=DEVICE
    )
    print(f"  Input images     : {dummy_images.shape}")
    print(f"  GT eye region    : {gt_eye_region.shape}")
    print(f"  GT visible iris  : {gt_visible_iris.shape}")
    print()

    # ---- Step 3: Forward pass ----
    print("[3/5] Running forward pass...")
    model.train()
    t0 = time.time()
    outputs = model(dummy_images)
    t_fwd = time.time() - t0

    print(f"  Forward time : {t_fwd:.3f}s")
    print()
    print("  Intermediate tensor shapes:")
    print(f"    predicted_eye_region_mask : {outputs['predicted_eye_region_mask'].shape}")
    print(f"    iris_params              : {outputs['iris_params'].shape}")
    print(f"    iris_params values (B=0) : {outputs['iris_params'][0].detach().cpu().tolist()}")
    print(f"    soft_iris_mask           : {outputs['soft_iris_mask'].shape}")
    print(f"    predicted_visible_iris   : {outputs['predicted_visible_iris'].shape}")
    print()

    # ---- Value range checks ----
    eye_mask = outputs["predicted_eye_region_mask"]
    iris_mask = outputs["soft_iris_mask"]
    vis_iris = outputs["predicted_visible_iris"]
    print("  Value ranges:")
    print(f"    eye_region_mask  : [{eye_mask.min().item():.4f}, {eye_mask.max().item():.4f}]")
    print(f"    soft_iris_mask   : [{iris_mask.min().item():.4f}, {iris_mask.max().item():.4f}]")
    print(f"    visible_iris     : [{vis_iris.min().item():.4f}, {vis_iris.max().item():.4f}]")
    print()

    # ---- Step 4: Compute loss ----
    print("[4/5] Computing loss...")
    losses = model.compute_loss(outputs, gt_eye_region, gt_visible_iris)
    print(f"  Loss_Eye   : {losses['loss_eye'].item():.6f}")
    print(f"  Loss_Iris  : {losses['loss_iris'].item():.6f}")
    print(f"  Total Loss : {losses['total_loss'].item():.6f}")
    print()

    # ---- Step 5: Backward pass (differentiability check) ----
    print("[5/5] Running loss.backward() to verify full differentiability...")
    t0 = time.time()
    losses["total_loss"].backward()
    t_bwd = time.time() - t0

    print(f"  Backward time : {t_bwd:.3f}s")
    print()

    # Verify gradients exist for key parameters
    grad_checks = {
        "backbone.encoder_features[0][0].weight": model.backbone.encoder_features[0][0].weight.grad,
        "iris_estimator.mlp[0].weight": model.iris_estimator.mlp[0].weight.grad,
        "backbone.seg_head[-1].weight": model.backbone.seg_head[-1].weight.grad,
    }

    all_ok = True
    for name, grad in grad_checks.items():
        has_grad = grad is not None and grad.abs().sum().item() > 0
        status = "✓ OK" if has_grad else "✗ MISSING"
        if not has_grad:
            all_ok = False
        print(f"  {status}  {name}")

    print()
    print("=" * 70)
    if all_ok:
        print("  ✓ ALL CHECKS PASSED — Pipeline is fully differentiable!")
    else:
        print("  ✗ SOME CHECKS FAILED — Review gradient flow.")
    print("=" * 70)


if __name__ == "__main__":
    main()
