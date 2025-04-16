import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetPlusPlus, self).__init__()

        # Encoder (Downsampling path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Decoder (Upsampling path)
        self.upconv1 = self.upconv_block(64, 128)
        self.upconv2 = self.upconv_block(128, 256)
        self.upconv3 = self.upconv_block(256, 512)

        # Final layer
        self.final_conv = nn.Conv2d(512, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder path with interpolation for upsampling
        up1 = F.interpolate(self.upconv1(enc1), size=enc2.shape[2:], mode='bilinear', align_corners=False) + enc2
        up2 = F.interpolate(self.upconv2(enc2), size=enc3.shape[2:], mode='bilinear', align_corners=False) + enc3
        up3 = F.interpolate(self.upconv3(enc3), size=enc4.shape[2:], mode='bilinear', align_corners=False) + enc4

        # Final output
        out = self.final_conv(up3)
        return out


