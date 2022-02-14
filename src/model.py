import lightly
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from sklearn.preprocessing import normalize

from utils import knn_predict


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback
    
    At the end of every training epoch we create a feature bank by inferencing
    the backbone on the dataloader passed to the module. 
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the 
    feature_bank features from the train data.
    We can access the highest accuracy during a kNN prediction using the 
    max_accuracy attribute.
    """

    def __init__(self, dataloader_kNN, cfg):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.gpus = 1 if torch.cuda.is_available() else 0
        self.classes = cfg.params.classes
        self.knn_k = cfg.params.knn_k
        self.knn_t = cfg.params.knn_t

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                if self.gpus > 0:
                    img = img.cuda()
                    target = target.cuda()
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, "feature_bank") and hasattr(self, "targets_bank"):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(
                feature,
                self.feature_bank,
                self.targets_bank,
                self.classes,
                self.knn_k,
                self.knn_t,
            )
            num = images.size(0)
            top1 = (pred_labels[:, 0] == targets).float().sum().item()
            return (num, top1)

    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_top1 = 0.0
            for (num, top1) in outputs:
                total_num += num
                total_top1 += top1
            acc = float(total_top1 / total_num)
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            self.log("epoch", self.current_epoch)
            self.log("kNN_accuracy", acc * 100.0, prog_bar=True)


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, cfg, gather_distributed=False):
        super().__init__(dataloader_kNN, cfg)

        # create a ResNet backbone and remove the classification head
        self.max_epochs = cfg.params.epoch_count
        self.cfg = cfg
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        print("Nice: ", hidden_dim)
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss(gather_distributed=gather_distributed)

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss, sync_dist=True)
        return loss

    def optimizer_steps(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=None,
        optimizer_closure=None,
        on_tpu=None,
        using_native_amp=None,
        using_lbfgs=None,
    ):
        # 120 steps ~ 1 epoch, implementing warmup
        if self.trainer.global_step < 1000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 1000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 1e-3

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.cfg.params.epoch_count
        )
        return [optim], [scheduler]


class SimCLRModel2(BenchmarkModule):
    def __init__(self, dataloader_kNN, cfg, gather_distributed=False):
        super().__init__(dataloader_kNN, cfg.params.classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        # create a simclr model based on ResNet
        self.cfg = cfg
        self.resnet_simclr = lightly.models.SimCLR(
            self.backbone, num_ftrs=cfg.params.num_ftrs
        )
        self.criterion = NTXentLoss(gather_distributed=gather_distributed)

    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(x0, x1)
        self.log("train_loss_ssl", loss)
        self.log("epoch", self.current_epoch)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_simclr.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.cfg.params.epoch_count
        )
        return [optim], [scheduler]
