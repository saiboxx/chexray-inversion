"""Classes and functions for encoder architectures."""
from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision.models import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)

RESNET_VARIANTS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
}

CONVNEXT_VARIANTS = {
    'convnext_tiny': convnext_tiny,
    'convnext_small': convnext_small,
    'convnext_base': convnext_base,
    'convnext_large': convnext_large,
}

# --------------------------------------------------------------------------------------
# ENCODER
# --------------------------------------------------------------------------------------


class ResnetEncoder(nn.Module):
    """Encoder based on ResNet."""

    def __init__(
        self,
        z_dim: int,
        in_size: int,
        resnet_type: str,
        pretrained: bool = False,
        gray_scale: bool = False,
    ) -> None:
        """Initialize ResNet decoder."""
        super().__init__()
        self.z_dim = z_dim
        self.in_size = in_size

        self.resnet_type = resnet_type
        self.model = RESNET_VARIANTS[resnet_type](pretrained=pretrained)

        pre_fc_size = self._get_pre_fc_size()

        if gray_scale:
            # Replace first conv to fit single channel inputs.
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False
            )

        # Replace last FC with 1x1 Kernel Conv Layer
        self.model.fc = nn.Conv2d(
            in_channels=pre_fc_size[0], out_channels=self.z_dim, kernel_size=(1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode a tensor to latent space."""
        features = self.features(x)
        return self.model.fc(features).squeeze(-1).squeeze(-1)

    def features(self, x: Tensor) -> Tensor:
        """Obtain hidden features of a tensor."""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x

    @torch.no_grad()
    def _get_pre_fc_size(self) -> Tuple[int, int, int]:
        """Compute shape of tensor before last fully connected layer."""
        x = torch.randn(1, 3, self.in_size, self.in_size)
        return self.features(x).squeeze(0).shape


class ConvNextEncoder(nn.Module):
    """Encoder based on ConvNext."""

    def __init__(
        self,
        z_dim: int,
        in_size: int,
        convnext_type: str,
        gray_scale: bool = False,
    ) -> None:
        """Initialize ResNet decoder."""
        super().__init__()
        self.z_dim = z_dim
        self.in_size = in_size

        self.convnext_type = convnext_type
        self.model = CONVNEXT_VARIANTS[convnext_type](num_classes=z_dim)

        if gray_scale:
            # Replace first conv to fit single channel inputs.
            self.model.features[0][0] = nn.Conv2d(
                1, 96, kernel_size=(4, 4), stride=(4, 4)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Encode a tensor to latent space."""
        return self.model(x)
