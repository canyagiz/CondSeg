"""
CondSeg Backbone & Eye-Region Segmenter
========================================
EfficientNet-B3 encoder with a U-Net style decoder that produces
a 1-channel eye-region segmentation mask.

The encoder extracts multi-scale feature maps at 5 levels.
The decoder progressively upsamples and fuses them via skip connections,
outputting a (B, 1, H, W) Sigmoid-activated probability map.

The deepest encoder feature map is also exposed for the IrisEstimator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


# ---------------------------------------------------------------------------
#  Decoder building blocks
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Single U-Net decoder block: upsample → concat skip → conv → BN → ReLU → conv → BN → ReLU.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # After concat the channel count is in_channels + skip_channels
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, in_channels, H, W)   — upsampled from previous decoder stage
            skip: (B, skip_channels, H', W')  — encoder skip connection (may differ in size)
        """
        # Bilinear upsample x to match skip's spatial dims
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)  # (B, in_ch + skip_ch, H', W')
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# ---------------------------------------------------------------------------
#  Backbone + Decoder
# ---------------------------------------------------------------------------

class BackboneWithDecoder(nn.Module):
    """
    EfficientNet-B3 encoder + U-Net decoder for eye-region segmentation.

    Outputs:
        predicted_eye_region_mask : (B, 1, H, W) — Sigmoid probability map
        deepest_features          : (B, C, h, w) — for IrisEstimator input

    EfficientNet-B3 feature map channels at each stage (after features block):
        Stage 0 (stem):   40 channels,  H/2
        Stage 1 (block0): 24 channels,  H/2
        Stage 2 (block1): 32 channels,  H/4
        Stage 3 (block2): 48 channels,  H/8
        Stage 4 (block3): 96 channels,  H/16
        Stage 5 (block4): 136 channels, H/16  (same spatial as stage 4)
        Stage 6 (block5): 232 channels, H/32
        Stage 7 (block6): 384 channels, H/32  (same spatial as stage 6)
        Stage 8 (block7): 1536 channels (after final conv+bn)  H/32

    We tap 5 levels for the U-Net skip architecture.
    """

    # EfficientNet-B3 feature indices and their output channels
    # We tap: after features[1] (24ch, H/2), features[2] (32ch, H/4),
    #         features[3] (48ch, H/8),  features[4] (96ch, H/16),
    #         features[8] (1536ch, H/32)  — deepest
    ENCODER_CHANNELS = [24, 32, 48, 96, 1536]  # channels at each tap point
    TAP_INDICES      = [1, 2, 3, 4, 8]         # indices into efficientnet features

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ---- Encoder: EfficientNet-B3 ----
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        # Extract the feature extraction layers (features is nn.Sequential of 9 blocks)
        self.encoder_features = backbone.features  # nn.Sequential with indices 0..8

        # ---- Decoder: 4 upsampling stages ----
        # Path:  deepest(1536) → up+skip4(96) → up+skip3(48) → up+skip2(32) → up+skip1(24)
        self.decoder4 = DecoderBlock(in_channels=1536, skip_channels=96,  out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256,  skip_channels=48,  out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128,  skip_channels=32,  out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64,   skip_channels=24,  out_channels=32)

        # ---- Segmentation head: 1-channel output with Sigmoid ----
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # 1-channel logit
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) — input image tensor

        Returns:
            predicted_eye_region_mask: (B, 1, H, W) — Sigmoid-activated eye-region map
            deepest_features:          (B, 1536, H/32, W/32) — for IrisEstimator
        """
        input_size = x.shape[2:]  # (H, W) — needed for final upsample

        # ---- Encoder forward: tap feature maps at each level ----
        features = {}
        out = x
        for i, layer in enumerate(self.encoder_features):
            out = layer(out)
            if i in self.TAP_INDICES:
                features[i] = out

        # Assign to named variables for clarity
        skip1 = features[1]   # (B, 24,  H/2,   W/2)
        skip2 = features[2]   # (B, 32,  H/4,   W/4)
        skip3 = features[3]   # (B, 48,  H/8,   W/8)
        skip4 = features[4]   # (B, 96,  H/16,  W/16)
        deepest = features[8] # (B, 1536, H/32, W/32)

        # ---- Decoder forward: progressively upsample with skip connections ----
        d4 = self.decoder4(deepest, skip4)  # (B, 256, H/16, W/16)
        d3 = self.decoder3(d4, skip3)       # (B, 128, H/8,  W/8)
        d2 = self.decoder2(d3, skip2)       # (B, 64,  H/4,  W/4)
        d1 = self.decoder1(d2, skip1)       # (B, 32,  H/2,  W/2)

        # ---- Segmentation head ----
        logits = self.seg_head(d1)  # (B, 1, H/2, W/2)

        # Upsample to original input resolution
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        # Apply Sigmoid to get probability map in [0, 1]
        predicted_eye_region_mask = torch.sigmoid(logits)  # (B, 1, H, W)

        return predicted_eye_region_mask, deepest
