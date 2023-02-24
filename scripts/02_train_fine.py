import argparse
import os
import signal
import sys
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch import nn
from torchvision.transforms import ToTensor

from src.chexray import ChexrayDataset
from src.losses.e4eloss import E4ELoss
from src.model.encoder import ConvNextEncoder, ResnetEncoder
from src.model.pgan import make_generator
from src.training import InverseFinetuneTrainer
from src.utils import read_yml


def run(rank: int, world_size: int, cfg: Dict) -> None:
    setup(rank, world_size)

    if rank == 0:
        wandb.init(project=cfg['WB']['project'], entity=cfg['WB']['entity'], config=cfg)

    # ----------------------------------------------------------------------------------
    # INITIALIZE ENCODER
    # ----------------------------------------------------------------------------------

    if 'resnet' in cfg['MODEL']['encoder_type']:
        model = ResnetEncoder(
            z_dim=512, in_size=1024, resnet_type=cfg['MODEL']['encoder_type']
        )
    else:
        model = ConvNextEncoder(
            z_dim=512, in_size=1024, convnext_type=cfg['MODEL']['encoder_type']
        )

    model = model.to(rank)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    enc_state_dict = \
    torch.load(cfg['MODEL']['encoder_checkpoint'], map_location=map_location)['model']
    enc_state_dict = {key.replace('module.', ''): val for key, val in
                      enc_state_dict.items()}
    model.load_state_dict(enc_state_dict)

    # ----------------------------------------------------------------------------------
    # INITIALIZE GENERATOR
    # ----------------------------------------------------------------------------------

    generator = make_generator(
        cfg['MODEL']['generator_path'], map_location='cuda:' + str(rank)
    )

    # ----------------------------------------------------------------------------------
    # INITIALIZE DATASETS
    # ----------------------------------------------------------------------------------
    train_transforms = ToTensor()

    test_transforms = ToTensor()

    train_dataset = ChexrayDataset(
        root=cfg['DATA']['data_path'],
        img_dir_name='images',
        train=True,
        transforms=train_transforms,
        rgb_convert=True
    )
    test_dataset = ChexrayDataset(
        root=cfg['DATA']['data_path'],
        img_dir_name='images',
        train=False,
        transforms=test_transforms,
        rgb_convert=True
    )

    # ----------------------------------------------------------------------------------
    # INITIALIZE LOSS FUNC
    # ----------------------------------------------------------------------------------

    loss_func = E4ELoss().to(rank)

    # ----------------------------------------------------------------------------------
    # INITIALIZE & RUN TRAINING
    # ----------------------------------------------------------------------------------

    trainer = InverseFinetuneTrainer(
        model=model,
        generator=generator,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss_func=loss_func,
        rank=rank,
        world_size=world_size,
        **cfg['TRAINER'],
    )

    trainer.train()

    cleanup()


def setup(rank: int, world_size: int) -> None:
    """Initialize environment for DDP training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    try:
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]
        default_port = int(default_port) + 15000  # type: ignore

    except Exception:
        default_port = 12910  # type: ignore

    os.environ['MASTER_PORT'] = str(default_port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    """Clean up process groups from DDP training."""
    wandb.finish()
    dist.destroy_process_group()


def signal_handler(signum, frame):
    cleanup()
    sys.exit()


def main() -> None:
    """Execute main func."""
    # Get correct config file over command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/inv_fine.yml', type=str)
    args = parser.parse_args()

    # Read yml file to memory as dict.
    cfg = read_yml(args.config)
    cfg['CFG_FILE'] = args.config

    # Initialize clean exit on SIGTERM
    signal.signal(signal.SIGTERM, signal_handler)

    # Start training with chosen configuration.
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, cfg), nprocs=world_size)


if __name__ == '__main__':
    main()
