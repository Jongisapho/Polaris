import json
import pytest
import torch

from sentiment_core.models import BiLSTMClassifier
from sentiment_core.models import BiLSTMSentimentModel
from sentiment_core.preprocessing import Vocabulary

from pathlib import Path

def write_tiny_checkpoint(
        directory: Path,
        vocab_size_in_config: int | None = None,
) -> Vocabulary:
    vocab = Vocabulary()
    vocab.build([
        "This is a great project",
        "This guy has terrible timing!",
        "I loved testing my skills",
        "Worst laptop ever haha !!!!"
        ],
        min_freq=1,
    )

    model = BiLSTMClassifier(vocab_size=len(vocab), embedding_dim=8, hidden_dim=16)
    torch.save(model.state_dict(), directory / "best_model.pt")
    vocab.save(directory / "vocab.json")

    config = {
        "embedding_dim" : 8,
        "hidden_dim" : 16,
        "vocab_size": len(vocab) if vocab_size_in_config is None else vocab_size_in_config
    }

    with open(directory / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    return vocab

    
def test_from_checkpoint_loads_and_predicts(tmp_path: Path) -> None:
    write_tiny_checkpoint(tmp_path)

    model = BiLSTMSentimentModel.from_checkpoint(tmp_path)
    label, confidence = model.predict("This project is great !")

    assert label in ("negative", "neutral", "positive")
    assert isinstance(confidence, float)
    assert 0.0 <= confidence < 1.0

def test_predict_is_deterministic(tmp_path: Path) -> None:
    write_tiny_checkpoint(tmp_path)
    model = BiLSTMSentimentModel.from_checkpoint(tmp_path)
    first = model.predict("worst experience ever")
    second = model.predict("worst experience ever")

    assert first == second

def test_from_checkpoint_rejects_mismatched_vocab(tmp_path: Path) -> None:
    vocab = write_tiny_checkpoint(tmp_path, vocab_size_in_config=None)
    write_tiny_checkpoint(tmp_path, vocab_size_in_config=len(vocab) + 5)
    with pytest.raises(ValueError, match="Vocabulary has"):
        BiLSTMSentimentModel.from_checkpoint(tmp_path)