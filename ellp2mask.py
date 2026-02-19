"""
CondSeg Ellp2Mask — Differentiable Ellipse-to-Mask Conversion
==============================================================
Converts batched 5D elliptical parameters (x0, y0, a, b, θ) into
soft segmentation masks using the general conic equation in matrix form.

All operations are pure PyTorch tensor ops — fully differentiable,
zero CPU bottleneck, GPU-friendly.

Mathematical Pipeline:
    1. Compute general conic coefficients A, B, C, D, E, F
    2. Construct 3×3 ellipse matrix M (off-diagonals divided by 2!)
    3. Compute distance map D = xᵀ M x  via torch.bmm
    4. Convert to soft mask S = Sigmoid( -D × effective_scale )

FIX: max(D) normalization removed — caused reward hacking at high res.
NOTE: effective_scale=10.0 provides resolution-independent gradients.
CRITICAL: M off-diagonal elements are B/2, D/2, E/2 (factor of 2).
"""

import torch
import torch.nn as nn
import math


class Ellp2Mask(nn.Module):
    """
    Differentiable module that converts 5D ellipse parameters to a soft mask.

    The coordinate grid is pre-computed and stored as a registered buffer
    (not a parameter) for performance — no gradient, moves with .to(device).

    Args:
        H:   Spatial height of the output mask (default 1024).
        W:   Spatial width of the output mask (default 1024).
    """

    def __init__(self, H: int = 1024, W: int = 1024):
        super().__init__()
        self.H = H
        self.W = W

        # ---- Pre-compute augmented coordinate grid ----
        # Pixel coordinates: y ∈ [0, H-1], x ∈ [0, W-1]
        # Augmented with 1 for homogeneous coords: [x, y, 1]ᵀ
        ys = torch.arange(H, dtype=torch.float32)  # (H,)
        xs = torch.arange(W, dtype=torch.float32)  # (W,)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each

        # Flatten and stack into augmented coordinates: (3, H*W)
        ones = torch.ones(H * W, dtype=torch.float32)
        coords = torch.stack([
            grid_x.reshape(-1),   # x-coordinates
            grid_y.reshape(-1),   # y-coordinates
            ones,                 # homogeneous 1
        ], dim=0)  # (3, H*W)

        # Register as buffer — not a parameter, but moves to GPU with model
        self.register_buffer("coords", coords)  # (3, H*W)

    def forward(self, params: torch.Tensor):
        """
        Args:
            params: (B, 5) — absolute ellipse parameters [x0, y0, a, b, θ]
                    x0, y0 in pixel coords; a, b in pixels; θ in radians

        Returns:
            soft_mask: (B, 1, H, W) — sigmoid-activated soft segmentation mask
            logits:    (B, 1, H, W) — raw logits before sigmoid (for BCEWithLogitsLoss)
        """
        B = params.shape[0]

        # ---- Unpack parameters ----
        x0    = params[:, 0]  # (B,) center x
        y0    = params[:, 1]  # (B,) center y
        a     = params[:, 2]  # (B,) semi-major axis
        b     = params[:, 3]  # (B,) semi-minor axis
        theta = params[:, 4]  # (B,) rotation angle

        # ---- Precompute trigonometric terms ----
        sin_t = torch.sin(theta)   # (B,)
        cos_t = torch.cos(theta)   # (B,)
        sin2  = sin_t ** 2         # (B,)
        cos2  = cos_t ** 2         # (B,)

        # Reciprocal of squared axes (avoid division later)
        a2_inv = 1.0 / (a ** 2 + 1e-8)   # (B,)  — tiny eps for numerical safety
        b2_inv = 1.0 / (b ** 2 + 1e-8)   # (B,)

        # ---- General conic coefficients ----
        # A = sin²θ / b² + cos²θ / a²
        A = sin2 * b2_inv + cos2 * a2_inv  # (B,)

        # B = 2(1/a² - 1/b²) sinθ cosθ
        B_coeff = 2.0 * (a2_inv - b2_inv) * sin_t * cos_t  # (B,)

        # C = cos²θ / b² + sin²θ / a²
        C = cos2 * b2_inv + sin2 * a2_inv  # (B,)

        # D = -2Ax₀ - By₀
        D_coeff = -2.0 * A * x0 - B_coeff * y0  # (B,)

        # E = -Bx₀ - 2Cy₀
        E_coeff = -B_coeff * x0 - 2.0 * C * y0  # (B,)

        # F = -(Dx₀ + Ey₀)/2 - 1
        F_coeff = -(D_coeff * x0 + E_coeff * y0) / 2.0 - 1.0  # (B,)

        # ---- Construct 3×3 Ellipse Matrix M ----
        # CRITICAL: Off-diagonal elements are HALVED
        # M = [[  A,   B/2, D/2 ],
        #      [ B/2,  C,   E/2 ],
        #      [ D/2,  E/2, F   ]]
        M = torch.zeros(B, 3, 3, device=params.device, dtype=params.dtype)
        M[:, 0, 0] = A
        M[:, 0, 1] = B_coeff / 2.0
        M[:, 0, 2] = D_coeff / 2.0
        M[:, 1, 0] = B_coeff / 2.0
        M[:, 1, 1] = C
        M[:, 1, 2] = E_coeff / 2.0
        M[:, 2, 0] = D_coeff / 2.0
        M[:, 2, 1] = E_coeff / 2.0
        M[:, 2, 2] = F_coeff
        # M shape: (B, 3, 3)

        # ---- Compute distance map D = xᵀ M x ----
        # coords: (3, H*W) — shared across batch, expand to (B, 3, H*W)
        coords_batch = self.coords.unsqueeze(0).expand(B, -1, -1)  # (B, 3, H*W)

        # Step 1: M @ coords → (B, 3, H*W)
        Mx = torch.bmm(M, coords_batch)  # (B, 3, H*W)

        # Step 2: xᵀ (M x) → element-wise multiply + sum over dim=1
        # This computes the quadratic form for each pixel
        dist_map = (coords_batch * Mx).sum(dim=1)  # (B, H*W)

        # Reshape to spatial dimensions
        dist_map = dist_map.view(B, self.H, self.W)  # (B, H, W)

        # ---- Convert distance map to soft mask ----
        # FIX: Makaledeki max(D)'ye bölme işlemi 1024x1024'te reward hacking'e
        # yol açıyordu — ağ, elipsi şişirerek d_max'ı düşürüp sahte beyaz maske
        # üretmeyi öğreniyordu. Bunun yerine sabit bir ölçek katsayısı kullanıyoruz.
        # Quadratic form D zaten eksen uzunluklarına (a, b) göre normalize olduğu
        # için sabit scale, gradyanların doğrudan elips sınırlarına akmasını sağlar.
        effective_scale = 10.0
        logits = -dist_map * effective_scale  # (B, H, W)

        soft_mask = torch.sigmoid(logits)  # (B, H, W)

        # Add channel dimension: (B, H, W) → (B, 1, H, W)
        logits = logits.unsqueeze(1)      # (B, 1, H, W)
        soft_mask = soft_mask.unsqueeze(1)  # (B, 1, H, W)

        return soft_mask, logits
