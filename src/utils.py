import lightly
import torch
import torchvision
from sklearn.preprocessing import normalize


def get_transforms(input_size):
    """
    This function returns the collate function for training transforms and
    the test transform.
    """
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size, gaussian_blur=0,
    )

    # We create a torchvision transformation for embedding the dataset after
    # training
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize["mean"],
                std=lightly.data.collate.imagenet_normalize["std"],
            ),
        ]
    )
    return collate_fn, test_transforms


def get_data(cfg):
    """
    This function returns the respective dataloaders.
    """
    collate_fn, test_transforms = get_transforms(cfg.params.input_size)
    dataset_train_simclr = lightly.data.LightlyDataset(input_dir=cfg.files.train_path)
    dataset_train_kNN = lightly.data.LightlyDataset(
        input_dir=cfg.files.train_path, transform=test_transforms
    )

    dataset_test = lightly.data.LightlyDataset(
        input_dir=cfg.files.test_path, transform=test_transforms
    )

    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=cfg.params.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=cfg.params.num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.params.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.params.num_workers,
    )

    print(cfg.params.batch_size)
    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=cfg.params.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.params.num_workers,
    )
    return dataloader_train_simclr, dataloader_test, dataloader_train_kNN


def knn_predict(
    feature, feature_bank, feature_labels, classes: int, knn_k: int, knn_t: float
):
    """Helper method to run kNN predictions on features based on a feature bank
    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: 
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
