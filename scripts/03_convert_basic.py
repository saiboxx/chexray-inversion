import os
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.chexray import Chexray2LatentDataset
from src.model.encoder import ConvNextEncoder, ResnetEncoder
from src.utils import save_dicts_to_csv

cfg: Dict = {
    'data_path': '/projects/data/chex-ray14',
    'encoder_type': 'convnext_small',
    'encoder_checkpoint': 'results_convnext_fine/model.pt',
    'device': 'cuda',
    'batch_size': 16,
    'num_workers': 1,
    'result_dir': 'result_latent',
}


def run(cfg: Dict) -> None:
    device = cfg['device']

    # ----------------------------------------------------------------------------------
    # INITIALIZE ENCODER
    # ----------------------------------------------------------------------------------
    if 'resnet' in cfg['encoder_type']:
        model = ResnetEncoder(z_dim=512, in_size=1024, resnet_type=cfg['encoder_type'])
    else:
        model = ConvNextEncoder(
            z_dim=512, in_size=1024, convnext_type=cfg['encoder_type']
        )

    model = model.to(device)
    enc_state_dict = torch.load(cfg['encoder_checkpoint'], map_location=device)['model']
    enc_state_dict = {
        key.replace('module.', ''): val for key, val in enc_state_dict.items()
    }
    model.load_state_dict(enc_state_dict)

    # ----------------------------------------------------------------------------------
    # INITIALIZE DATASETS
    # ----------------------------------------------------------------------------------
    train_transforms = ToTensor()

    test_transforms = ToTensor()

    train_dataset = Chexray2LatentDataset(
        root=cfg['data_path'],
        img_dir_name='images',
        train=True,
        transforms=train_transforms,
        rgb_convert=True,
    )
    test_dataset = Chexray2LatentDataset(
        root=cfg['data_path'],
        img_dir_name='images',
        train=False,
        transforms=test_transforms,
        rgb_convert=True,
    )

    # ----------------------------------------------------------------------------------
    # START CONVERSION TO LATENT SPACE
    # ----------------------------------------------------------------------------------
    os.makedirs(cfg['result_dir'], exist_ok=True)
    # I know there are better solutions
    path_stem = cfg['result_dir'] + '/' + cfg['result_stem']

    print('---> Converting Training Set <---')
    latent_codes, meta = convert_to_latent(
        encoder=model,
        dataset=train_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        device=device,
    )

    torch.save(latent_codes, path_stem + '_train_z.pt')
    save_dicts_to_csv(meta, path_stem + '_train_meta.csv')

    print('---> Converting Test Set <---')
    latent_codes, meta = convert_to_latent(
        encoder=model,
        dataset=test_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        device=device,
    )

    torch.save(latent_codes, path_stem + '_test_z.pt')
    save_dicts_to_csv(meta, path_stem + '_test_meta.csv')


@torch.no_grad()
def convert_to_latent(
    encoder: nn.Module,
    dataset: Chexray2LatentDataset,
    batch_size: int,
    num_workers: int,
    device: Union[torch.device, str, int],
) -> Tuple[Tensor, List[Dict]]:
    encoder.to(device)
    encoder.eval()

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=Chexray2LatentDataset.collate_fn,
    )

    latent_codes: List[Tensor] = []
    metas: List[Dict] = []
    for imgs, meta in tqdm(data_loader):
        imgs = imgs.to(device)
        zs = encoder(imgs)

        latent_codes.append(zs.detach().cpu())
        metas.extend(meta)

    return torch.cat(latent_codes), metas


def main() -> None:
    """Execute main func."""
    # Start conversion to latent space with chosen configuration.
    run(cfg)


if __name__ == '__main__':
    main()
