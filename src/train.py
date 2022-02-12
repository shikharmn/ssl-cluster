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
os.environ["HYDRA_FULL_ERROR"] = "1"  # For config related stack-trace
wandb.init(project="ssl-cluster", entity="shikhar_mn")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: CIFARConfig) -> None:
    wandb.config = cfg
    gpus = 1 if torch.cuda.is_available() else 0
    dataloader_train_simclr, dataloader_test, dataloader_kNN_train = get_data(cfg)
    model = SimCLRModel(dataloader_kNN_train, cfg)
    wandb_logger = WandbLogger(project="ssl-cluster")

    trainer = pl.Trainer(
        max_epochs=cfg.params.epoch_count,
        gpus=1,
        fast_dev_run=True,
        logger=wandb_logger,
    )
    trainer.fit(
        model,
        train_dataloaders=dataloader_train_simclr,
        val_dataloaders=dataloader_test,
    )


if __name__ == "__main__":
    main()
