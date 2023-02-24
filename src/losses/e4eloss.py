from torch import Tensor, nn

from src.losses.lpips import LPIPS
from src.losses.moco_loss import MocoLoss


class E4ELoss(nn.Module):
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_moco: float = 0.1,
        lambda_lpips: float = 0.8,
        moco_path: str = 'models/mocov2.pt',
        lpips_net: str = 'alex',
    ) -> None:
        super().__init__()
        self.lambda_lpips = lambda_lpips
        self.lambda_moco = lambda_moco
        self.lambda_mse = lambda_mse

        self.mse_loss = nn.MSELoss()
        self.moco_loss = MocoLoss(moco_path)
        self.lpips_loss = LPIPS(net_type=lpips_net)
        self.lpips_loss.eval()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        loss = self.mse_loss(x_hat, x) * self.lambda_mse
        if self.lambda_moco > 0.0:
            loss += self.moco_loss(x_hat, x) * self.lambda_moco
        if self.lambda_lpips > 0.0:
            loss += self.lpips_loss(x_hat, x, normalize=True) * self.lambda_lpips
        return loss
