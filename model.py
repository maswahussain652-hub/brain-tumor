"""
Complete 3D UNet (UNet3D) Neural Network Architecture in PyTorch.
Designed for volumetric 3D medical image segmentation with residual connections,
Instance Normalization, and multi-scale feature skip concatenation.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CONFIG


class ConvBlock3D(nn.Module):
    """
    Dual 3D Convolution block with Instance Normalization, LeakyReLU,
    and a residual shortcut connection to maintain gradient propagation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        leaky_slope: float = 0.1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)

        self.dropout = (
            nn.Dropout3d(dropout_prob) if dropout_prob > 0 else nn.Identity()
        )

        # Residual projection shortcut if channels change
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        out = self.act2(out + res)
        return out


class DownBlock3D(nn.Module):
    """Downsampling stage: MaxPool3d followed by ConvBlock3D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        leaky_slope: float = 0.1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            leaky_slope=leaky_slope,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pooled = self.pool(x)
        return self.conv(x_pooled)


class UpBlock3D(nn.Module):
    """
    Upsampling stage: Transposed 3D convolution, feature concatenation
    with skip connection, and ConvBlock3D.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        leaky_slope: float = 0.1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock3D(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            leaky_slope=leaky_slope,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)

        # Handle potential odd spatial dimensions
        diff_d = skip.size(2) - x_up.size(2)
        diff_h = skip.size(3) - x_up.size(3)
        diff_w = skip.size(4) - x_up.size(4)

        if diff_d > 0 or diff_h > 0 or diff_w > 0:
            x_up = F.pad(
                x_up,
                (
                    diff_w // 2,
                    diff_w - diff_w // 2,
                    diff_h // 2,
                    diff_h - diff_h // 2,
                    diff_d // 2,
                    diff_d - diff_d // 2,
                ),
            )

        cat_features = torch.cat([skip, x_up], dim=1)
        return self.conv(cat_features)


class UNet3D(nn.Module):
    """
    Production-grade 3D UNet for Volumetric Brain Lesion Segmentation.
    Processes volumetric input tensors of shape (Batch, Channels, Depth, Height, Width).
    """

    def __init__(
        self,
        in_channels: int = MODEL_CONFIG.IN_CHANNELS,
        out_channels: int = MODEL_CONFIG.OUT_CHANNELS,
        base_filters: int = MODEL_CONFIG.BASE_FILTERS,
        dropout_prob: float = MODEL_CONFIG.DROPOUT_PROB,
        leaky_slope: float = MODEL_CONFIG.LEAKY_SLOPE,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters

        # Encoder Path
        f1 = base_filters        # e.g. 16
        f2 = base_filters * 2    # e.g. 32
        f3 = base_filters * 4    # e.g. 64
        f4 = base_filters * 8    # e.g. 128 (Bottleneck)

        self.input_block = ConvBlock3D(
            in_channels, f1, leaky_slope=leaky_slope, dropout_prob=dropout_prob
        )
        self.down1 = DownBlock3D(
            f1, f2, leaky_slope=leaky_slope, dropout_prob=dropout_prob
        )
        self.down2 = DownBlock3D(
            f2, f3, leaky_slope=leaky_slope, dropout_prob=dropout_prob
        )
        self.bottleneck = DownBlock3D(
            f3, f4, leaky_slope=leaky_slope, dropout_prob=dropout_prob
        )

        # Decoder Path
        self.up2 = UpBlock3D(
            in_channels=f4,
            skip_channels=f3,
            out_channels=f3,
            leaky_slope=leaky_slope,
            dropout_prob=dropout_prob,
        )
        self.up1 = UpBlock3D(
            in_channels=f3,
            skip_channels=f2,
            out_channels=f2,
            leaky_slope=leaky_slope,
            dropout_prob=dropout_prob,
        )
        self.up0 = UpBlock3D(
            in_channels=f2,
            skip_channels=f1,
            out_channels=f1,
            leaky_slope=leaky_slope,
            dropout_prob=dropout_prob,
        )

        # 1x1x1 Final Convolutional Classification Head
        self.classifier = nn.Conv3d(
            f1,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        # Initialize network weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits of shape (B, OutChannels, D, H, W).
        """
        # Encoder
        x0 = self.input_block(x)       # (B, f1, D, H, W)
        x1 = self.down1(x0)            # (B, f2, D/2, H/2, W/2)
        x2 = self.down2(x1)            # (B, f3, D/4, H/4, W/4)
        b = self.bottleneck(x2)        # (B, f4, D/8, H/8, W/8)

        # Decoder with Skip Connections
        d2 = self.up2(b, x2)           # (B, f3, D/4, H/4, W/4)
        d1 = self.up1(d2, x1)          # (B, f2, D/2, H/2, W/2)
        d0 = self.up0(d1, x0)          # (B, f1, D, H, W)

        logits = self.classifier(d0)
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities for binary segmentation in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def create_unet3d(device: Optional[torch.device] = None) -> UNet3D:
    """Factory helper to instantiate UNet3D on selected device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    model.eval()
    return model
