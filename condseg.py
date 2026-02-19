"""
CondSeg Model — Wrapper & Loss
===============================
Connects the Backbone+Decoder (eye-region segmenter), IrisEstimator,
and Ellp2Mask into a single end-to-end differentiable pipeline.

Training forward pass:
    1. Encode image → eye-region mask + deepest features
    2. Estimate iris ellipse params from deepest features
    3. Ellp2Mask → soft iris mask
    4. Conditioned Assemble: predicted_visible_iris = soft_iris_mask × eye_region_mask
    5. Loss = BCE(eye_region, GT) + BCE(visible_iris, GT)

CRITICAL: Uses nn.BCELoss (NOT BCEWithLogitsLoss) because all masks
have already been passed through Sigmoid activations.
"""

import torch
import torch.nn as nn
from torch.amp import autocast

from backbone import BackboneWithDecoder
from iris_estimator import IrisEstimator
from ellp2mask import Ellp2Mask


class CondSeg(nn.Module):
    """
    Full CondSeg pipeline: image → ellipse parameters + segmentation masks.

    Args:
        img_size:    Input image spatial size (assumes square, default 1024).
        tau:         Ellp2Mask temperature (default 800.0).
        epsilon:     Minimum relative axis length for iris (default 0.01).
        pretrained:  Whether to use pretrained EfficientNet-B3 weights.
    """

    def __init__(
        self,
        img_size: int = 1024,
        tau: float = 800.0,
        epsilon: float = 0.01,
        pretrained: bool = True,
    ):
        super().__init__()
        self.img_size = img_size

        # ---- Sub-modules ----
        # 1. Backbone encoder + U-Net decoder for eye-region segmentation
        self.backbone = BackboneWithDecoder(pretrained=pretrained)

        # 2. Iris regression head (takes deepest encoder features)
        self.iris_estimator = IrisEstimator(
            in_channels=1536,  # EfficientNet-B3 final feature channels
            hidden_dim=256,
            dropout=0.2,
            epsilon=epsilon,
        )

        # 3. Differentiable ellipse → soft mask conversion
        self.ellp2mask = Ellp2Mask(H=img_size, W=img_size, tau=tau)

        # ---- Loss function ----
        # CRITICAL: BCELoss, NOT BCEWithLogitsLoss — outputs already have Sigmoid
        self.bce_loss = nn.BCELoss()

    def forward(self, images: torch.Tensor):
        """
        Forward pass (inference mode).

        Args:
            images: (B, 3, H, W) — input image tensor

        Returns:
            dict with keys:
                'predicted_eye_region_mask': (B, 1, H, W) — Sigmoid probability
                'soft_iris_mask':            (B, 1, H, W) — differentiable ellipse mask
                'predicted_visible_iris':    (B, 1, H, W) — conditioned assemble result
                'iris_params':               (B, 5)       — absolute [x0, y0, a, b, θ]
        """
        B, C, H, W = images.shape

        # ---- Step 1: Backbone → eye-region mask + deepest features ----
        predicted_eye_region_mask, deepest_features = self.backbone(images)
        # predicted_eye_region_mask: (B, 1, H, W) — already Sigmoid-activated
        # deepest_features:          (B, 1536, h, w)

        # ---- Step 2: Iris Estimator → 5D absolute ellipse params ----
        iris_params = self.iris_estimator(deepest_features, H, W)
        # iris_params: (B, 5) — [x0, y0, a, b, θ]

        # ---- Step 3: Ellp2Mask → soft iris mask ----
        # CRITICAL: Force float32 to prevent AMP float16 overflow.
        # Pixel coords up to 1024 → x² ≈ 1,046,529 > float16 max (65504)
        with autocast('cuda', enabled=False):
            soft_iris_mask = self.ellp2mask(iris_params.float())
        # soft_iris_mask: (B, 1, H, W)

        # ---- Step 4: Conditioned Assemble ----
        # Visibility of iris is conditioned on the eye-region being open.
        # CRITICAL: We detach() the eye mask here to decouple training.
        # This prevents the iris loss from "fighting" the eye loss and
        # destroying the eye mask to reduce iris error.
        predicted_visible_iris = soft_iris_mask * predicted_eye_region_mask.detach()
        # predicted_visible_iris: (B, 1, H, W)

        return {
            "predicted_eye_region_mask": predicted_eye_region_mask,
            "soft_iris_mask": soft_iris_mask,
            "predicted_visible_iris": predicted_visible_iris,
            "iris_params": iris_params,
        }

    def compute_loss(
        self,
        outputs: dict,
        gt_eye_region: torch.Tensor,
        gt_visible_iris: torch.Tensor,
    ) -> dict:
        """
        Compute the CondSeg training loss.

        Args:
            outputs:         dict from forward() with predicted masks
            gt_eye_region:   (B, 1, H, W) float32 — binary GT (sclera|iris|pupil)
            gt_visible_iris: (B, 1, H, W) float32 — binary GT (iris|pupil)

        Returns:
            dict with keys:
                'loss_eye':   scalar — BCE loss for eye-region segmentation
                'loss_iris':  scalar — BCE loss for conditioned iris segmentation
                'total_loss': scalar — sum of both losses
        """
        # ---- Cast to float32 for BCELoss (unsafe under AMP autocast) ----
        pred_eye  = outputs["predicted_eye_region_mask"].float()
        pred_iris = outputs["predicted_visible_iris"].float()
        gt_eye_f  = gt_eye_region.float()
        gt_iris_f = gt_visible_iris.float()

        # ---- Loss_Eye = BCE(predicted_eye_region, GT_Eye_Region) ----
        loss_eye = self.bce_loss(pred_eye, gt_eye_f)

        # ---- Loss_Iris = BCE only INSIDE eye-region (per paper Sec. 3.1) ----
        # Paper: "eye-region mask is used as an ignorance mask, where only
        # pixels inside the eye-region calculate loss and back-propagate
        # gradients, while the ones outside eye-region are regarded as ignored."
        eye_mask = gt_eye_f  # (B, 1, H, W) — binary mask
        num_eye_pixels = eye_mask.sum().clamp(min=1.0)

        # Manual BCE with masking (nn.BCELoss doesn't support per-pixel masking)
        bce_per_pixel = -(gt_iris_f * torch.log(pred_iris + 1e-7)
                          + (1 - gt_iris_f) * torch.log(1 - pred_iris + 1e-7))
        loss_iris = (bce_per_pixel * eye_mask).sum() / num_eye_pixels

        # ---- Total Loss ----
        total_loss = loss_eye + loss_iris

        return {
            "loss_eye": loss_eye,
            "loss_iris": loss_iris,
            "total_loss": total_loss,
        }
