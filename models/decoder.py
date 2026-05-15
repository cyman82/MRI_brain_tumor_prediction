import torch
import torch.nn as nn

from models.image_encoder import DoubleConv3D


# =========================================================
# 3D DECODER
# =========================================================

class Decoder3D(nn.Module):

    def __init__(
        self,
        features=[32, 64, 128, 256],
        out_channels=4
    ):

        super(Decoder3D, self).__init__()

        self.ups = nn.ModuleList()

        reversed_features = features[::-1]

        # -------------------------------------------------
        # UPSAMPLING BLOCKS
        # -------------------------------------------------

        for feature in reversed_features:

            self.ups.append(

                nn.ConvTranspose3d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2
                )

            )

            self.ups.append(

                DoubleConv3D(
                    feature * 2,
                    feature
                )

            )

        # -------------------------------------------------
        # FINAL CONVOLUTION
        # -------------------------------------------------

        self.final_conv = nn.Conv3d(
            features[0],
            out_channels,
            kernel_size=1
        )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, x, skip_connections):

        skip_connections = skip_connections[::-1]

        # -------------------------------------------------
        # DECODER
        # -------------------------------------------------

        for idx in range(0, len(self.ups), 2):

            # ---------------------------------------------
            # UPSAMPLE
            # ---------------------------------------------

            x = self.ups[idx](x)

            skip_connection = skip_connections[idx // 2]

            # ---------------------------------------------
            # CONCATENATE SKIP CONNECTION
            # ---------------------------------------------

            x = torch.cat(
                (skip_connection, x),
                dim=1
            )

            # ---------------------------------------------
            # DOUBLE CONV
            # ---------------------------------------------

            x = self.ups[idx + 1](x)

        # -------------------------------------------------
        # FINAL OUTPUT
        # -------------------------------------------------

        return self.final_conv(x)