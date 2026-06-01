"""
utils/model.py
--------------
UNet3D architecture definition (single source of truth).

Architecture specification:
- 3 encoder levels + bottleneck, base features=16
- InstanceNorm3d + ReLU, MaxPool3d(2) downsampling
- ConvTranspose3d(2) upsampling with skip connections
- Final layer: concat(decoder1_output, encoder1_output)
  -> Conv3d(features*2, 1, kernel_size=1) -> sigmoid
- Input shape: (B, 1, 160, 224, 256)

Used by: steps/s1_train/train.py, steps/s2_inference/inference.py
"""

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: int = 16):
        super().__init__()
        # Encoder
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._block(features * 4, features * 8)

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        # Multi-scale fusion: concat(decoder1_output, encoder1_output) -> features*2 channels
        self.conv = nn.Conv3d(features * 2, out_channels, kernel_size=1)

    def _block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))
        # Bottleneck
        x4 = self.bottleneck(self.pool3(x3))
        # Decoder
        d3 = self.decoder3(torch.cat((self.upconv3(x4), x3), dim=1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), x2), dim=1))
        d1 = self.decoder1(torch.cat((self.upconv1(d2), x1), dim=1))
        # Multi-scale fusion
        fusion = torch.cat((d1, x1), dim=1)
        return torch.sigmoid(self.conv(fusion))
