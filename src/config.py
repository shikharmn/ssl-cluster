from dataclasses import dataclass


@dataclass
class Paths:
    log: str


@dataclass
class Files:
    train_path: str
    test_path: str


@dataclass
class Params:
    epoch_count: int
    input_size : int
    num_ftrs : int
    batch_size : int
    num_workers : int
    knn_k : int
    knn_t : float
    classes : int


@dataclass
class CIFARConfig:
    paths: Paths
    files: Files
    params: Params
