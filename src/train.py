import os
import sys

import hydra
import lightly
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from click import progressbar
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from config import CIFARConfig
from model import SimCLRModel
from utils import get_data, get_transforms

sys.path.insert(0, "../")
cs = ConfigStore.instance()
cs.store(name="cifar_config", node=CIFARConfig)
os.environ["HYDRA_FULL_ERROR"] = "1"  # For config related stack-trace


@hydra.main(config_path="conf", config_name="config")
def main(cfg: CIFARConfig) -> None:

    gpus = 1 if torch.cuda.is_available() else 0
    dataloader_train_simclr, dataloader_test, dataloader_kNN_train = get_data(cfg)
    model = SimCLRModel(dataloader_kNN_train, gpus, cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.params.epoch_count, gpus=gpus, fast_dev_run=True
    )
    trainer.fit(
        model,
        train_dataloaders=dataloader_train_simclr,
        val_dataloaders=dataloader_test,
    )


if __name__ == "__main__":
    main()
