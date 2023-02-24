import argparse
import os
import signal
import sys
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch.nn import L1Loss

from src.model.encoder import ConvNextEncoder, ResnetEncoder
from src.model.pgan import make_generator
from src.training import InverseBoostrapTrainer
from src.utils import read_yml


def run(rank: int, world_size: int, cfg: Dict) -> None:
    setup(rank, world_size)

    if rank == 0:
        wandb.init(project=cfg['WB']['project'], entity=cfg['WB']['entity'], config=cfg)

    if 'resnet' in cfg['MODEL']['encoder_type']:
        model = ResnetEncoder(
            z_dim=512, in_size=1024, resnet_type=cfg['MODEL']['encoder_type']
        )
    else:
        model = ConvNextEncoder(
            z_dim=512, in_size=1024, convnext_type=cfg['MODEL']['encoder_type']
        )

    generator = make_generator(
        cfg['MODEL']['generator_path'], map_location='cuda:' + str(rank)
    )

    loss_func = L1Loss()

    trainer = InverseBoostrapTrainer(
        model=model,
        generator=generator,
        loss_func=loss_func,
        rank=rank,
        world_size=world_size,
        **cfg['TRAINER'],
    )

    if cfg['TRAINER']['start_from_checkpoint']:
        trainer.load_checkpoint(checkpoint_path=cfg['TRAINER']['old_checkpoint'])

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
    parser.add_argument('-c', '--config', default='configs/inv_boot.yml', type=str)
    parser.add_argument('-s', '--stint', default=0, type=int)
    args = parser.parse_args()

    # Read yml file to memory as dict.
    cfg = read_yml(args.config)
    cfg['CFG_FILE'] = args.config

    # Check if training is part of a multi-stint run
    if args.stint != 0:
        if args.stint == 1:
            cfg['TRAINER']['start_from_checkpoint'] = False
        else:
            cfg['TRAINER']['start_from_checkpoint'] = True
            cfg['TRAINER']['old_checkpoint'] = os.path.join(
                cfg['TRAINER']['checkpoint_dir'], 'v' + str(args.stint - 1), 'model.pt'
            )

        cfg['TRAINER']['checkpoint_dir'] = os.path.join(
            cfg['TRAINER']['checkpoint_dir'], 'v' + str(args.stint)
        )
        os.makedirs(cfg['TRAINER']['checkpoint_dir'], exist_ok=True)

    else:
        cfg['TRAINER']['start_from_checkpoint'] = False

    # Initialize clean exit on SIGTERM
    signal.signal(signal.SIGTERM, signal_handler)

    # Start training with chosen configuration.
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, cfg), nprocs=world_size)


if __name__ == '__main__':
    main()
