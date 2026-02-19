"""
CondSeg Iris Estimator (Regression Head)
=========================================
Takes the deepest encoder feature map, applies Global Average Pooling
and an MLP to regress 5D elliptical parameters for the full iris.

Output: normalized parameters in (0, 1) via Sigmoid, then converted
to absolute pixel coordinates.

Conversion (exactly per user specification):
    x0 = x̂0 × W
    y0 = ŷ0 × H
    a  = (â + ε) × min(W, H) / 2
    b  = (b̂ + ε) × min(W, H) / 2
    θ  = θ̂ × π              (maps [0,1] → [0, π])

    ε = 0.01 for iris (prevents degenerate zero-length axes)
"""

import math
import torch
import torch.nn as nn


class IrisEstimator(nn.Module):
    """
    Regression head: encoder features → 5D ellipse parameters.

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

        # Global Average Pooling collapses spatial dims → (B, C, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # MLP: C → hidden → hidden → 5
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
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

        # ---- Global Average Pooling ----
        x = self.gap(features)        # (B, C, 1, 1)
        x = x.view(B, -1)            # (B, C)

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
        a  = (a_hat + self.epsilon) * min_dim / 2.0     # semi-major axis (pixels)
        b  = (b_hat + self.epsilon) * min_dim / 2.0     # semi-minor axis (pixels)
        theta = t_hat * math.pi                         # angle in [0, π] radians

        # Stack back to (B, 5)
        params_abs = torch.stack([x0, y0, a, b, theta], dim=1)  # (B, 5)

        return params_abs
