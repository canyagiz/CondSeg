"""
CondSeg Iris Estimator (Spatial Regression Head)
=================================================
Takes the deepest encoder feature map and regresses 5D elliptical
parameters using spatial-aware strided convolutions instead of GAP.

WHY NOT GAP?
    GAP collapses the entire 32×32 feature map into a single 1×1 pixel,
    destroying all spatial (X, Y) coordinate information. The model knows
    "there is an iris" but cannot tell WHERE it is — leading to random
    center predictions and degenerate elongated ellipses.

SOLUTION: Spatial Compression
    Use strided Conv2d layers to gradually reduce the feature map while
    PRESERVING spatial structure, then pool to a 4×4 grid (16 spatial
    bins). The MLP can then learn position-dependent mappings.

Output: normalized parameters in (0, 1) via Sigmoid, then converted
to absolute pixel coordinates.

Conversion:
    x0 = x̂0 × W
    y0 = ŷ0 × H
    a  = (â + ε) × min(W, H) / 4
    b  = (b̂ + ε) × min(W, H) / 4
    θ  = θ̂                   (raw sigmoid output, range [0, 1])

    ε = 0.01 for iris (prevents degenerate zero-length axes)
"""

import math
import torch
import torch.nn as nn


class IrisEstimator(nn.Module):
    """
    Spatial regression head: encoder features → 5D ellipse parameters.

    Instead of Global Average Pooling (which destroys spatial info),
    uses strided convolutions to compress features while retaining
    a 4×4 spatial grid — giving the model 16 spatial bins to
    localize the iris center.

    Args:
        in_channels: Number of channels in the deepest encoder feature map (1536 for EfficientNet-B3).
        hidden_dim:  Hidden layer width in the MLP.
        dropout:     Dropout probability between MLP layers.
        epsilon:     Minimum relative axis length to prevent degenerate ellipses.
    """

    def __init__(
        self,
        in_channels: int = 1536,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.epsilon = epsilon

        # ---- Spatial Compression (replaces GAP) ----
        # Gradually reduce channels while preserving spatial structure.
        # Input: (B, 1536, 32, 32) → Output: (B, 128, 4, 4)
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=2),  # → (B, 256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=2),          # → (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),                                      # → (B, 128, 4, 4)
        )

        # 4×4 grid × 128 channels = 2048-dim flattened vector
        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),   # 5D output: (x̂, ŷ, â, b̂, θ̂)
        )

        # Sigmoid constrains all outputs to (0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            features: (B, C, h, w) — deepest encoder feature map
            H:        Original input image height (e.g. 1024)
            W:        Original input image width  (e.g. 1024)

        Returns:
            params_abs: (B, 5) — absolute ellipse parameters [x0, y0, a, b, θ]
                        in pixel coordinates / radians
        """
        B = features.shape[0]

        # ---- Spatial Compression (preserves 4×4 spatial grid) ----
        x = self.spatial_compress(features)   # (B, 128, 4, 4)
        x = x.view(B, -1)                    # (B, 2048) — spatial info preserved!

        # ---- MLP + Sigmoid → normalized params in (0, 1) ----
        params_norm = self.sigmoid(self.mlp(x))  # (B, 5)

        # Unpack normalized predictions
        x_hat = params_norm[:, 0]     # (B,)  center x, normalized
        y_hat = params_norm[:, 1]     # (B,)  center y, normalized
        a_hat = params_norm[:, 2]     # (B,)  semi-major, normalized
        b_hat = params_norm[:, 3]     # (B,)  semi-minor, normalized
        t_hat = params_norm[:, 4]     # (B,)  angle θ, normalized

        # ---- Convert to absolute coordinates ----
        min_dim = min(H, W)

        x0 = x_hat * W                                 # pixel x-center
        y0 = y_hat * H                                 # pixel y-center
        a  = (a_hat + self.epsilon) * min_dim / 4.0     # semi-axis (pixels), /4 caps max radius
        b  = (b_hat + self.epsilon) * min_dim / 4.0     # semi-axis (pixels), /4 caps max radius
        theta = t_hat                                   # angle (raw sigmoid, [0, 1])

        # Stack back to (B, 5)
        params_abs = torch.stack([x0, y0, a, b, theta], dim=1)  # (B, 5)

        return params_abs
