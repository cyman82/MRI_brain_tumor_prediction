import torch
import torch.nn as nn


# =========================================================
# DOUBLE CONVOLUTION BLOCK
# =========================================================

class DoubleConv3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(DoubleConv3D, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


# =========================================================
# 3D IMAGE ENCODER
# =========================================================

class ImageEncoder3D(nn.Module):

    def __init__(self, in_channels=1, features=[32, 64, 128, 256]):

        super(ImageEncoder3D, self).__init__()

        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # -------------------------------------------------
        # ENCODER BLOCKS
        # -------------------------------------------------

        for feature in features:

            self.downs.append(
                DoubleConv3D(
                    in_channels,
                    feature
                )
            )

            in_channels = feature

        # -------------------------------------------------
        # BOTTLENECK
        # -------------------------------------------------

        self.bottleneck = DoubleConv3D(
            features[-1],
            features[-1] * 2
        )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, x):

        skip_connections = []

        # -------------------------------------------------
        # ENCODER
        # -------------------------------------------------

        for down in self.downs:

            x = down(x)

            skip_connections.append(x)

            x = self.pool(x)

        # -------------------------------------------------
        # BOTTLENECK
        # -------------------------------------------------

        x = self.bottleneck(x)

        return x, skip_connections