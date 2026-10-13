import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from sentiment_core.architecture import BiLSTMClassifier
from sentiment_core.labels import LABELS
from sentiment_core.preprocessing import Vocabulary


@dataclass
class TrainingConfig:
    embedding_dim: int = 100
    hidden_dim: int = 128
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 15
    patience: int = 3
    grad_clip: float = 1.0
    use_class_weights: bool = True
    seed: int = 2026

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TweetDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocabulary) -> None:
        self.x = torch.tensor([vocab.encode(t) for t in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

def make_loader(dataset: Dataset, batch_size: int, shuffle:bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)