import os
import sys

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.config_store import ConfigStore
from pytorch_lightning.loggers import WandbLogger

from config import CIFARConfig
from model import SimCLRModel
from utils import get_data

sys.path.insert(0, "../")
cs = ConfigStore.instance()
cs.store(name="cifar_config", node=CIFARConfig)
# os.environ["HYDRA_FULL_ERROR"] = "1"  # For config related stack-trace
wandb.init(project="ssl-cluster", entity="shikhar_mn")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: CIFARConfig) -> None:
    wandb.config = cfg
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    distributed = False
    sync_batchnorm, gather_distributed = distributed, distributed

    if distributed:
        distributed_backend = "ddp"
        batch_sizes = cfg.params.batch_size // gpus
    else:
        distributed_backend = None
        gpus = min(gpus, 1)

    dataloader_train_simclr, dataloader_test, dataloader_kNN_train = get_data(
        cfg, gpus=gpus
    )
    model = SimCLRModel(dataloader_kNN_train, cfg, gather_distributed)
    wandb.watch(model, log_freq=100)
    wandb_logger = WandbLogger(project="ssl-cluster")

    # check_path = "/home/shikhar-ug/Projects/ssl-cluster/outputs/2022-02-13/13-49-12/None/version_None/checkpoints/epoch=60-step=5916.ckpt"

    trainer = pl.Trainer(
        max_epochs=cfg.params.epoch_count,
        gpus=gpus,
        strategy=distributed_backend,
        sync_batchnorm=sync_batchnorm,
        fast_dev_run=False,
        logger=wandb_logger,
        precision=16,
    )
    trainer.fit(
        model,
    #    ckpt_path=check_path,
        train_dataloaders=dataloader_train_simclr,
        val_dataloaders=dataloader_test,
    )


if __name__ == "__main__":
    main()
