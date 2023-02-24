import os
from typing import (
    Dict,
    Optional,
    Union,
)

import torch
import wandb
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
)
from torchmetrics import MeanMetric
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm


class InverseBoostrapTrainer:
    def __init__(
        self,
        model: nn.Module,
        generator: nn.Module,
        learning_rate: float,
        weight_decay: float,
        loss_func: nn.Module,
        steps: int,
        batch_size: int,
        checkpoint_interval: int,
        checkpoint_dir: str,
        validation_interval: int,
        verbose_interval: int,
        rank: int,
        world_size: int,
        fp_16: bool,
        *args,
        **kwargs
    ) -> None:
        self.model = model.to(rank)
        self.model = DDP(self.model, device_ids=[rank])

        self.generator = generator.to(rank)
        if fp_16:
            self.generator.half()
        self.generator.eval()

        self.optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=Adam,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.loss_func = loss_func

        self.batch_size = batch_size
        self.steps = steps

        self.validation_interval = validation_interval
        self.verbose_interval = verbose_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_step = 1
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.scaler = GradScaler()
        self.fp_16 = fp_16

        self.rank = rank
        self.world_size = world_size

    def train(self) -> None:
        # ------------------------------------------------------------------------------
        # START TRAINING PROCEDURE
        # ------------------------------------------------------------------------------
        self.model.train()

        loss_hist = []
        for step in range(self.checkpoint_step, self.steps + 1):

            batch_loss = self.train_step()
            self.optimize(batch_loss)

            if self.rank == 0:
                loss_hist.append(float(batch_loss))

                if step % self.verbose_interval == 0:
                    mean_loss = sum(loss_hist) / len(loss_hist)
                    loss_hist.clear()
                    print('STEP: {:3} | LOSS: {:.5f}'.format(step, mean_loss))

                if step % self.checkpoint_interval == 0:
                    self.save_checkpoint(step)
                    print('---> CHECKPOINT SAVED <---')

                if step % self.validation_interval == 0:
                    self.model.eval()
                    reconstruction_samples = self.sample_grid()
                    wandb.log(
                        {
                            'step': step,
                            'train/loss': mean_loss,
                            'general/reconstruction': wandb.Image(
                                to_pil_image(reconstruction_samples)
                            ),
                        }
                    )

    def optimize(self, batch_loss: Tensor) -> None:
        self.optimizer.zero_grad()

        if self.fp_16:
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            batch_loss.backward()
            self.optimizer.step()

    def train_step(self) -> Tensor:
        with torch.no_grad():
            z = torch.randn(
                self.batch_size,
                512,
                dtype=torch.half if self.fp_16 else torch.float,
                device='cuda:' + str(self.rank),
            )
            x = self.generator(z)
            x = torch.clamp(x, 0, 1).detach()

        with autocast(enabled=self.fp_16):
            z_hat = self.model(x)

        # Exclude loss func from autocast due to instability with resulting NaNs
        batch_loss = self.loss_func(z_hat, z)

        return batch_loss

    @torch.no_grad()
    def sample_grid(self) -> Tensor:
        z = torch.randn(
            8,
            512,
            dtype=torch.half if self.fp_16 else torch.float,
            device='cuda:' + str(self.rank),
        )
        x = self.generator(z)
        x.clamp_(0, 1)

        z_hat = self.model.module(x.float())

        x_hat = self.generator(z_hat.half())
        x_hat.clamp_(0, 1)
        grid = make_grid(
            torch.cat([x.float().detach().cpu(), x_hat.float().detach().cpu()]), nrow=8
        )
        return grid

    def save_checkpoint(self, step: Optional[int] = None) -> None:
        data = {
            'step': step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(data, os.path.join(self.checkpoint_dir, 'model.pt'))

    def load_checkpoint(self, checkpoint_path) -> None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.checkpoint_step = state_dict['step']
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

        print(
            'Loaded checkpoint from "{}" with step id {}'.format(
                checkpoint_path, state_dict['step']
            )
        )

        del state_dict


class InverseFinetuneTrainer:
    def __init__(
        self,
        model: nn.Module,
        generator: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_workers: int,
        learning_rate: float,
        weight_decay: float,
        loss_func: nn.Module,
        epochs: int,
        batch_size: int,
        checkpoint_dir: str,
        rank: int,
        world_size: int,
        fp_16: bool,
        *args,
        **kwargs
    ) -> None:
        self.model = model.to(rank)
        self.model = DDP(self.model, device_ids=[rank])

        self.generator = generator.to(rank)
        if fp_16:
            self.generator.half()
        for p in self.generator.parameters():
            p.requires_grad = False
        self.generator.eval()

        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.loss_func = loss_func

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.scaler = GradScaler()
        self.fp_16 = fp_16

        self.rank = rank
        self.world_size = world_size

        self.metric_agent = MetricAgent(rank)

    def get_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    def train(self) -> None:
        train_loader = self.get_dataloader(self.train_dataset)
        test_loader = self.get_dataloader(self.test_dataset, shuffle=False)

        # ----------------------------------------------------------------------------------
        # START TRAINING PROCEDURE
        # ----------------------------------------------------------------------------------
        for ep in range(1, self.epochs + 1):
            train_loader.sampler.set_epoch(ep)

            # ------------------------------------------------------------------------------
            # TRAINING LOOP
            # ------------------------------------------------------------------------------
            self.model.train()
            for batch_data in tqdm(train_loader, leave=False):
                batch_loss = self.train_step(batch_data)
                self.optimize(batch_loss)

                self.metric_agent.collect(batch_loss, train=True)

            # ------------------------------------------------------------------------------
            # TEST LOOP
            # ------------------------------------------------------------------------------
            self.model.eval()
            for batch_data in tqdm(test_loader, leave=False):
                batch_loss = self.test_step(batch_data)

                self.metric_agent.collect(batch_loss, train=False)

            ep_metrics = self.metric_agent.compute()
            self.metric_agent.reset()

            if self.rank == 0:
                # ---------------------------------------
                # PRINT PROGRESS
                # ---------------------------------------
                print(
                    'EP: {:3} | LOSS: T {:.5f} V {:.5f}'.format(
                        ep, ep_metrics['train/loss'], ep_metrics['test/loss']
                    )
                )

                # ---------------------------------------
                # SAVE MODEL CHECKPOINT
                # ---------------------------------------
                self.save_checkpoint(ep)
                print('---> CHECKPOINT SAVED <---')

                # ---------------------------------------
                # LOG METRICS TO WANDB
                # ---------------------------------------
                self.model.eval()
                reconstruction_samples = self.sample_grid()
                ep_metrics.update(
                    {
                        'ep': ep,
                        'general/reconstruction': wandb.Image(
                            to_pil_image(reconstruction_samples)
                        ),
                    }
                )
                wandb.log(ep_metrics)

    def optimize(self, batch_loss: Tensor) -> None:
        self.optimizer.zero_grad()

        if self.fp_16:
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            batch_loss.backward()
            self.optimizer.step()

    def train_step(self, batch_data: Dict) -> Tensor:
        x = batch_data['image'].to(self.rank)

        with autocast(enabled=self.fp_16):
            z_hat = self.model(x)
            x_hat = self.generator(z_hat)
            x_hat.clamp_(0, 1)

        # Exclude loss func from autocast due to instability with resulting NaNs
        batch_loss = self.loss_func(x_hat.float(), x)

        return batch_loss

    @torch.no_grad()
    def test_step(self, batch_data: Dict) -> Tensor:
        return self.train_step(batch_data)

    @torch.no_grad()
    def sample_grid(self) -> Tensor:
        z = torch.randn(
            8,
            512,
            dtype=torch.half if self.fp_16 else torch.float,
            device='cuda:' + str(self.rank),
        )
        x = self.generator(z)
        x.clamp_(0, 1)

        z_hat = self.model.module(x.float())

        x_hat = self.generator(z_hat.half())
        x_hat.clamp_(0, 1)
        grid = make_grid(
            torch.cat([x.float().detach().cpu(), x_hat.float().detach().cpu()]), nrow=8
        )
        return grid

    def save_checkpoint(self, ep: Optional[int] = None) -> None:
        data = {
            'ep': ep,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(data, os.path.join(self.checkpoint_dir, 'model.pt'))


class MetricAgent:
    def __init__(self, device: Union[torch.device, int]):
        self.train_loss = MeanMetric(compute_on_step=False).to(device)
        self.test_loss = MeanMetric(compute_on_step=False).to(device)

        self.device = device

    def collect(self, loss: Tensor, train: bool) -> None:
        if train:
            self.train_loss(loss)
        else:
            self.test_loss(loss)

    def compute(self) -> Dict:
        return {
            'train/loss': float(self.train_loss.compute()),
            'test/loss': float(self.test_loss.compute()),
        }

    def reset(self):
        self.train_loss.reset()
        self.test_loss.reset()
