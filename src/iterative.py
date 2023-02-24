import os
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.optim import Adam
from tqdm import trange

from src.utils import get_lr


class IterativeOptimizer:
    def __init__(
        self,
        generator: nn.Module,
        learning_rate: float = 0.1,
        steps: int = 1000,
        loss_func: nn.Module = nn.MSELoss(),
        optimizer_class: type = Adam,
        eps: float = 0.0,
    ) -> None:
        self.generator = generator
        for p in self.generator.parameters():
            p.requires_grad = False
        self.generator.eval()

        self.learning_rate = learning_rate
        self.steps = steps
        self.eps = eps
        self.loss_func = loss_func

        self.optimizer_class = optimizer_class

    def run(self, z: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:

        z = z.clone()
        z.requires_grad = True
        optimizer = self.optimizer_class([z], lr=self.learning_rate)

        for i in (pbar := trange(self.steps)):

            t = i / self.steps
            lr = get_lr(t, self.learning_rate, ramp_down=0.4)
            optimizer.param_groups[0]['lr'] = lr

            x_hat = self.generator(z).clamp(0, 1)
            loss = self.loss_func(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (loss < self.eps).all():
                break

            if i % 100 == 0:
                pbar.set_description('Loss: {:.4f}'.format(float(loss)))

        z.requires_grad = False

        return z, x_hat


def collect_iterative_results(root: str) -> Tensor:
    files = os.listdir(root)

    # Check if there are missing files
    files_complete = True
    batch_idxs = sorted([int(f.replace('.pt', '')) for f in files])
    for i in range(len(batch_idxs) - 1):
        if batch_idxs[i+1] - batch_idxs[i] > 1:
            files_complete = False
            print('There is a gap between {} and {}'.format(batch_idxs[i], batch_idxs[i+1]))
    assert files_complete, 'There a missing files! Collection aborted.'

    result_dict = {}
    for f in files:
        result_dict.update(torch.load(os.path.join(root, f), map_location='cpu'))

    result_vals = []
    for i in range(len(result_dict)):
        try:
            result_vals.append(result_dict[i])
        except KeyError:
            print('KeyError occurred! Missing Entry with Index {}'.format(i))

    return torch.stack(result_vals)
