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


@dataclass
class CIFARConfig:
    paths: Paths
    files: Files
    params: Params
