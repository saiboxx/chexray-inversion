import argparse
import os
from typing import Dict

import pandas as pd
import torch
from torch.optim import Adam

from src.iterative import IterativeOptimizer
from src.losses.e4eloss import E4ELoss
from src.model.pgan import make_generator
from src.utils import load_images

cfg: Dict = {
    'data_path': '/projects/data/chex-ray14/images',
    'train_meta_path': 'results_latent/chexray_latent_train_meta.csv',
    'train_latent_path': 'results_latent/chexray_latent_train_z.pt',
    'test_meta_path': 'results_latent/chexray_latent_test_meta.csv',
    'test_latent_path': 'results_latent/chexray_latent_test_z.pt',
    'generator_path': 'models/chexgan_generator.pt',
    'device': 'cuda',
    'batch_size': 32,
    'steps': 3000,
    'result_dir': 'results_latent',
}


def run(cfg: Dict, is_train: bool, batch_idx: int) -> None:
    device = cfg['device']

    print('STARTING OPTIM FOR BATCH IDX {}'.format(batch_idx))
    # ----------------------------------------------------------------------------------
    # FIND CORRECT ENTRIES
    # ----------------------------------------------------------------------------------
    meta_path = cfg['train_meta_path'] if is_train else cfg['test_meta_path']
    z_path = cfg['train_latent_path'] if is_train else cfg['test_latent_path']

    meta = pd.read_csv(meta_path)

    min_idx = batch_idx * cfg['batch_size']
    max_idx = min(min_idx + cfg['batch_size'], len(meta))

    if min_idx > len(meta) - 1:
        return

    meta = pd.read_csv(meta_path)[min_idx:max_idx]
    file_ids = meta['file'].tolist()

    # Get initial z vals from encoder stage
    zs = torch.load(z_path, map_location=device)[min_idx:max_idx]
    # Load image labels
    file_paths = [os.path.join(cfg['data_path'], f) for f in file_ids]
    xs = load_images(file_paths).to(device)

    # ----------------------------------------------------------------------------------
    # INITIALIZE GENERATOR
    # ----------------------------------------------------------------------------------
    generator = make_generator(
        state_dict_path=cfg['generator_path'], map_location=device
    )
    generator = generator.to(device)
    for p in generator.parameters():
        p.requires_grad = False
    generator.eval()

    # ----------------------------------------------------------------------------------
    # START ITERATIVE OPTIMIZATION
    # ----------------------------------------------------------------------------------
    iter_opt = IterativeOptimizer(
        generator=generator,
        learning_rate=0.1,
        steps=cfg['steps'],
        loss_func=E4ELoss(lambda_moco=0.1).to(device),
        optimizer_class=Adam,
    )

    new_z, _ = iter_opt.run(zs, xs)

    # ----------------------------------------------------------------------------------
    # SAVE RESULT
    # ----------------------------------------------------------------------------------
    target_dir = os.path.join(
        cfg['result_dir'], 'optim_' + ('train' if is_train else 'test')
    )
    os.makedirs(target_dir, exist_ok=True)

    result_dict = {}
    for i in range(len(new_z)):
        result_dict.update({i + min_idx: new_z[i]})

    path = os.path.join(target_dir, str(batch_idx) + '.pt')
    torch.save(result_dict, path)
    print('SUCCESSFULLY SAVED in "{}".'.format(path))


def main() -> None:
    """Execute main func."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default=0, type=int)
    parser.add_argument('-s', '--set', default='train', type=str)
    args = parser.parse_args()

    # Start conversion to latent space with chosen configuration.
    is_train = 'train' in args.set
    if is_train:
        print('TRAINING SET CHOSEN!')
    else:
        print('TEST SET CHOSEN!')

    run(cfg, is_train, args.index - 1)


if __name__ == '__main__':
    main()
