"""
CondSeg Model — Wrapper & Loss
===============================
Connects the Backbone+Decoder (eye-region segmenter), IrisEstimator,
and Ellp2Mask into a single end-to-end differentiable pipeline.

Training forward pass:
    1. Encode image → eye-region mask + logits + deepest features
    2. Estimate iris ellipse params from deepest features
    3. Ellp2Mask → soft iris mask + logits
    4. Conditioned Assemble: predicted_visible_iris = soft_iris_mask × eye_region_mask
    5. Loss = BCE+Dice(eye_logits, GT) + masked BCE+Dice(iris_logits, GT)

CRITICAL: Uses F.binary_cross_entropy_with_logits on raw logits (not
sigmoid outputs) for BCE term. Dice Loss uses sigmoid probabilities for
boundary-focused overlap optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from backbone import BackboneWithDecoder
from iris_estimator import IrisEstimator
from ellp2mask import Ellp2Mask


class CondSeg(nn.Module):
    """
    Full CondSeg pipeline: image → ellipse parameters + segmentation masks.

    Args:
        img_size:    Input image spatial size (assumes square, default 1024).
        tau:         (Removed — Ellp2Mask now uses fixed effective_scale)
        epsilon:     Minimum relative axis length for iris (default 0.01).
        pretrained:  Whether to use pretrained EfficientNet-B3 weights.
    """

    def __init__(
        self,
        img_size: int = 1024,
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
        self.ellp2mask = Ellp2Mask(H=img_size, W=img_size)

        # ---- Loss function ----
        # Using F.binary_cross_entropy_with_logits (computed on raw logits)
        # instead of nn.BCELoss to prevent vanishing gradients from
        # sigmoid saturation (sigmoid(large) = 1.0 → gradient = 0).

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

        # ---- Step 1: Backbone → eye-region mask + logits + deepest features ----
        predicted_eye_region_mask, eye_region_logits, deepest_features = self.backbone(images)
        # predicted_eye_region_mask: (B, 1, H, W) — Sigmoid-activated
        # eye_region_logits:         (B, 1, H, W) — raw logits (for loss)
        # deepest_features:          (B, 1536, h, w)

        # ---- Step 2: Iris Estimator → 5D absolute ellipse params ----
        iris_params = self.iris_estimator(deepest_features, H, W)
        # iris_params: (B, 5) — [x0, y0, a, b, θ]

        # ---- Step 3: Ellp2Mask → soft iris mask + logits ----
        # CRITICAL: Force float32 to prevent AMP float16 overflow.
        # Pixel coords up to 1024 → x² ≈ 1,046,529 > float16 max (65504)
        with autocast('cuda', enabled=False):
            soft_iris_mask, iris_logits = self.ellp2mask(iris_params.float())
        # soft_iris_mask: (B, 1, H, W)
        # iris_logits:    (B, 1, H, W) — raw logits (for loss)

        # ---- Step 4: Conditioned Assemble ----
        # Visibility of iris is conditioned on the eye-region being open.
        # CRITICAL: detach() decouples training — iris loss cannot destroy eye mask.
        predicted_visible_iris = soft_iris_mask * predicted_eye_region_mask.detach()
        # predicted_visible_iris: (B, 1, H, W)

        return {
            "predicted_eye_region_mask": predicted_eye_region_mask,
            "eye_region_logits": eye_region_logits,
            "soft_iris_mask": soft_iris_mask,
            "iris_logits": iris_logits,
            "predicted_visible_iris": predicted_visible_iris,
            "iris_params": iris_params,
        }

    @staticmethod
    def _dice_loss(pred_probs: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor = None, smooth: float = 1.0) -> torch.Tensor:
        """
        Dice Loss — sınır örtüşmesine odaklanır.

        Args:
            pred_probs: Sigmoid-activated predictions (B, 1, H, W)
            target:     Binary ground truth (B, 1, H, W)
            mask:       Optional spatial mask (B, 1, H, W) — only compute
                        Dice inside this region (e.g. eye-region for iris)
            smooth:     Laplace smoothing to avoid 0/0
        """
        if mask is not None:
            pred_probs = pred_probs * mask
            target = target * mask

        pred_flat = pred_probs.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1.0 - (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )

    def compute_loss(
        self,
        outputs: dict,
        gt_eye_region: torch.Tensor,
        gt_visible_iris: torch.Tensor,
    ) -> dict:
        """
        Compute the CondSeg training loss: BCE + Dice.

        BCE  → pixel-level accuracy (logits)
        Dice → region overlap / boundary precision (probabilities)

        Args:
            outputs:         dict from forward() with predicted masks and logits
            gt_eye_region:   (B, 1, H, W) float32 — binary GT (sclera|iris|pupil)
            gt_visible_iris: (B, 1, H, W) float32 — binary GT (iris|pupil)

        Returns:
            dict with keys:
                'loss_eye':   scalar — BCE + Dice for eye-region segmentation
                'loss_iris':  scalar — masked BCE + Dice for iris estimation
                'total_loss': scalar — sum of both losses
        """
        # ---- Get logits (NOT sigmoid outputs) for stable gradient flow ----
        eye_logits  = outputs["eye_region_logits"].float()
        iris_logits = outputs["iris_logits"].float()
        gt_eye_f    = gt_eye_region.float()
        gt_iris_f   = gt_visible_iris.float()

        # ---- Loss_Eye ----
        # BCE on logits
        bce_eye = F.binary_cross_entropy_with_logits(eye_logits, gt_eye_f)
        # Dice on probabilities
        eye_probs = torch.sigmoid(eye_logits)
        dice_eye = self._dice_loss(eye_probs, gt_eye_f)
        loss_eye = bce_eye + dice_eye

        # ---- Loss_Iris = masked (BCE + Dice) inside GT eye-region ----
        # Paper Sec. 3.1: "eye-region mask is used as an ignorance mask"
        eye_mask = gt_eye_f  # (B, 1, H, W) — binary condition mask
        num_eye_pixels = eye_mask.sum().clamp(min=1.0)

        # Masked BCE (logits)
        bce_iris = F.binary_cross_entropy_with_logits(
            iris_logits, gt_iris_f, weight=eye_mask, reduction='sum'
        ) / num_eye_pixels
        # Masked Dice (probabilities)
        iris_probs = torch.sigmoid(iris_logits)
        dice_iris = self._dice_loss(iris_probs, gt_iris_f, mask=eye_mask)
        loss_iris = bce_iris + dice_iris

        # ---- Anatomik Dairesellik Cezası (Circularity Loss) ----
        # İris 3D'de bir çemberdir; ekrana yansıdığında a ≈ b olmalı.
        # Model ignorance mask'ı exploit edip devasa uzun elipsler (a≠b)
        # üretmesin diye L1 cezası ekliyoruz.
        a_params = outputs["iris_params"][:, 2]  # yarı-eksen a
        b_params = outputs["iris_params"][:, 3]  # yarı-eksen b
        loss_shape = F.l1_loss(a_params, b_params)
        shape_weight = 0.1

        # ---- Total Loss ----
        total_loss = loss_eye + loss_iris + (loss_shape * shape_weight)

        return {
            "loss_eye": loss_eye,
            "loss_iris": loss_iris,
            "loss_shape": loss_shape,
            "total_loss": total_loss,
        }

