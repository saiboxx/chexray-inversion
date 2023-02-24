import csv
import math
import os
import pickle
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import matplotlib.pyplot as plt
import torch
import yaml  # type: ignore
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import Tensor
from torchvision.utils import make_grid
from PIL import Image
from torchvision.transforms.functional import to_tensor


def get_lr(t: float, initial_lr: float, ramp_down: float = 0.25, ramp_up: float = 0.05):
    lr_ramp = min(1.0, (1 - t) / ramp_down)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1.0, t / ramp_up)
    return initial_lr * lr_ramp


def plot_image(
        img: Tensor,
        fig_size: Tuple[int, int] = (10, 10),
        ncols: Optional[int] = None,
        show: bool = True,
        save_path: Optional[str] = None,
) -> None:
    img = img.detach().cpu()

    # Shape of 4 implies multiple image inputs
    if len(img.shape) == 4:
        img = make_grid(img, nrow=ncols if ncols is not None else len(img))

    plt.figure(figsize=fig_size)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


def plot_rec_grid(img_org: Tensor, img_enc: Tensor, img_opt: Tensor,
                  fig_size: Tuple[int, int] = (20, 10),
                  show: bool = True,
                  save_path: Optional[str] = None,
                  ) -> None:
    # Move all tensors to cpu and detach them
    img_org = img_org.detach().cpu()
    img_enc = img_enc.detach().cpu()
    img_opt = img_opt.detach().cpu()

    # Compute residuals from original image
    res_enc = torch.mean(torch.abs(img_org - img_enc), dim=0).clamp(0, 1)
    res_opt = torch.mean(torch.abs(img_org - img_opt), dim=0).clamp(0, 1)

    # Arrange Grid
    fig = plt.figure(figsize=fig_size)
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.0, )

    grid[0].imshow(img_org.permute(1, 2, 0).numpy())
    grid[1].imshow(img_enc.permute(1, 2, 0).numpy())
    grid[2].imshow(img_opt.permute(1, 2, 0).numpy())
    grid[3].imshow(torch.zeros_like(res_enc).numpy())
    grid[4].imshow(res_enc.numpy())
    grid[5].imshow(res_opt.numpy())

    for ax in grid:
        ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


def read_yml(filepath: str) -> Dict:
    """Load a yml file to memory as dict."""
    with open(filepath, 'r') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))


def read_file(file_path: str) -> List[str]:
    """Read a generic file line by line."""
    with open(file_path, 'r') as f:
        return [line.replace('\n', '') for line in f.readlines()]


def read_pickle(filepath: str) -> Any:
    """Load a binary file to a python object."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(file: Any, filepath: str) -> None:
    """
    Save a generic object to a binary file.

    :param file: Object to be saved
    :param filepath: Saving destination
    """
    file_dir, _ = os.path.split(filepath)
    os.makedirs(file_dir, exist_ok=True)

    with open(filepath, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_dicts_to_csv(file: List[Dict], file_path: str) -> None:
    with open(file_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=file[0].keys())
        writer.writeheader()
        for data in file:
            writer.writerow(data)


def load_image(file_path: str) -> Tensor:
    img = Image.open(file_path).convert('RGB')
    return to_tensor(img)


def load_images(file_paths: List[str]) -> Tensor:
    return torch.stack([load_image(p) for p in file_paths])
