from typing import Union

import torch
from torch import Tensor, nn
from torchvision.models import vgg16
from torchvision.transforms import Normalize


class VGGPerceptualLoss(nn.Module):
    """
    VGG perceptual loss module.

    The loss compares inner activations of VGG16 trained on image-net.
    """

    def __init__(
        self,
        mean_reduction: bool = True,
        device: Union[str, int] = 'cpu',
        normalize: bool = True,
        to_rgb: bool = False,
    ):
        """Initialize perceptual loss."""
        super().__init__()
        vgg = vgg16(pretrained=True).to(device)
        vgg.eval()

        blocks = [
            vgg.features[:4],
            vgg.features[4:9],
            vgg.features[9:16],
            vgg.features[16:23],
        ]

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.reduction = 'mean' if mean_reduction else 'sum'
        self.loss_func = nn.L1Loss(reduction=self.reduction)

        self.do_to_rgb = to_rgb
        self.do_normalize = normalize
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """Compute the loss for two tensors."""
        loss = torch.zeros(1, dtype=x.dtype, device=x.device)

        if self.do_to_rgb:
            x = x.expand(-1, 3, -1, -1)
            x_hat = x_hat.expand(-1, 3, -1, -1)

        if self.do_normalize:
            x = self.normalize(x)
            x_hat = self.normalize(x_hat)

        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                x = block(x)
            x_hat = block(x_hat)
            loss += self.loss_func(x, x_hat)

        if self.reduction == 'sum':
            return loss
        else:
            return loss / len(self.blocks)
