import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class DecoderBlock(nn.Module):
    """UNet decoder block with upsampling and convolution."""

    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """ResNet50-UNet for semantic segmentation."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Load ResNet50 encoder
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)

        # Encoder layers (freeze if needed)
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1  # 256 channels
        self.encoder_layer2 = resnet.layer2  # 512 channels
        self.encoder_layer3 = resnet.layer3  # 1024 channels
        self.encoder_layer4 = resnet.layer4  # 2048 channels

        # Decoder
        self.decoder4 = DecoderBlock(2048, 1024, skip_channels=1024)  # layer4 + layer3
        self.decoder3 = DecoderBlock(1024, 512, skip_channels=512)    # decoder4 + layer2
        self.decoder2 = DecoderBlock(512, 256, skip_channels=256)     # decoder3 + layer1
        self.decoder1 = DecoderBlock(256, 128, skip_channels=64)      # decoder2 + conv1

        # Final segmentation head
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.encoder_conv1(x)
        conv1 = self.encoder_bn1(conv1)
        conv1 = self.encoder_relu(conv1)

        pool1 = self.encoder_maxpool(conv1)
        layer1 = self.encoder_layer1(pool1)  # [B, 256, H/4, W/4]
        layer2 = self.encoder_layer2(layer1)  # [B, 512, H/8, W/8]
        layer3 = self.encoder_layer3(layer2)  # [B, 1024, H/16, W/16]
        layer4 = self.encoder_layer4(layer3)  # [B, 2048, H/32, W/32]

        # Decoder with skip connections
        dec4 = self.decoder4(layer4, layer3)  # [B, 1024, H/16, W/16]
        dec3 = self.decoder3(dec4, layer2)    # [B, 512, H/8, W/8]
        dec2 = self.decoder2(dec3, layer1)    # [B, 256, H/4, W/4]
        dec1 = self.decoder1(dec2, conv1)     # [B, 128, H/2, W/2]

        # Final prediction (logits)
        out = self.final_conv(dec1)  # [B, num_classes, H/2, W/2]

        # Upsample to original resolution (assuming input H=W, can be adjusted)
        out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

        return out

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze decoder and final conv
        for module in [self.decoder4, self.decoder3, self.decoder2, self.decoder1, self.final_conv]:
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_encoder(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def create_resnet_unet(num_classes, pretrained=True):
    """Factory function to create ResNet50-UNet model."""
    return ResNetUNet(num_classes, pretrained)
