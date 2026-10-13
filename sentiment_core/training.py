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
class TrainConfig:
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

def compute_class_weights(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    return counts.sum() / (counts * num_classes)

def train_one_epoch(model, loader, loss_fn, optimizer, device, grad_clip) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_examples += len(y)

    return total_loss / total_examples

def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * len(y)
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(y.cpu().tolist())

    n = len(all_true)
    per_class = f1_score(all_true, all_pred, average=None, labels=[0, 1, 2])

    return {
        "loss": total_loss / n,
        "accuracy": sum(t == p for t, p in zip(all_true, all_pred)) / n,
        "macro_f1": f1_score(all_true, all_pred, average="macro"),
        "per_class_f1":[float(v) for v in per_class]
    }

def fit(model, train_loader, val_loader, loss_fn, config: TrainConfig, device, checkpoint_dir: str | Path) -> list[dict]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_f1 = -1.0
    wait = 0
    history: list[dict] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, config.grad_clip)
        val = evaluate(model, val_loader, loss_fn, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **val
        })
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | "
              f"val loss {val['loss']:.4f} | val acc {val['accuracy']:.3f} | "
              f"val macro-F1 {val['macro_f1']:.3f}")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            wait = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= config.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    return history


def train(train_data, val_data, config: TrainConfig, checkpoint_dir="checkpoints"):
    set_seed(config.seed)
    device = get_device()
    train_texts, train_labels = train_data
    val_texts, val_labels = val_data

    vocab = Vocabulary()
    vocab.build(train_texts)

    train_loader = make_loader(TweetDataset(train_texts, train_labels, vocab), config.batch_size
                               , shuffle=True)
    val_loader = make_loader(TweetDataset(val_texts, val_labels, vocab), 256, shuffle=False)

    model = BiLSTMClassifier(vocab_size=len(vocab), embedding_dim=config.embedding_dim,
                             hidden_dim=config.hidden_dim).to(device)
    weights = compute_class_weights(train_labels) if config.use_class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    history = fit(model, train_loader, val_loader, loss_fn, config, device, checkpoint_dir)

    vocab.save(Path(checkpoint_dir) / "vocab.json")
    with open(Path(checkpoint_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(config), "vocab_size": len(vocab)}, f, indent=2)
    return history