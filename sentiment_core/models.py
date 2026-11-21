from abc import ABC, abstractmethod # abstractclass in python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
import torch.nn.functional as F
from sentiment_core.architecture import BiLSTMClassifier
from sentiment_core.preprocessing import Vocabulary
from sentiment_core.labels import LABELS
from pathlib import Path
import json



class SentimentModel(ABC):
    """
    Abstract interface all sentiment models will implement.
    Guarantess a consistent predict() signature regardless of the 
    underlying model.
    """

    @abstractmethod
    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict sentiment for a single piece of text.

        Args:
            text(str): raw input text (unprocessed)

        Returns:
            tuple[Str, float]: (label, confidence) where labels is one of "positive",
            "negative" or "neutral" and confidence is a float in 
            [-1.0, 1.0] depending on the model.
        """
        raise NotImplementedError("Subclass must implement predict() method.")

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """
        Predict sentiment for a list of texts
        Default implementation calls predict() in a loop
        subclass may override for a faster vectorized path
        """
        return [self.predict(text) for text in texts]

class VaderSentimentModel(SentimentModel):
    """
    Rule-Based sentiment model using VADER (Valence Aware Dictionary and Sentiment Reasoner)
    No Training and Preprocessing
    Operates on the raw text
    """

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def predict(self, text:str) -> tuple[str, float]:
        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return label, compound

class BiLSTMSentimentModel(SentimentModel):
    """
    Deep learning sentiment model using a bidirectional LSTM.
    Requires a build vocabularry and (eventually) trained weights.
    """

    def __init__(self, vocab: "Vocabulary", embedding_dim: int = 100, hidden_dim: int = 128, weights_path: str | Path | None = None) -> None:
        self._vocab = vocab
        self._model = BiLSTMClassifier(vocab_size=len(vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim)

        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)

        self._model.eval()

    @classmethod
    def from_checkpoint(cls, directory: str | Path) -> "BiLSTMSentimentModel":
        directory = Path(directory)

        with open(directory / "config.json", encoding="utf-8") as f:
            config = json.load(f)

        vocab  =  Vocabulary.load(directory / "vocab.json")
        if len(vocab) != config["vocab_size"]:
            raise ValueError(
                f"Vocabulary has {len(vocab)} entries but the checkpoint was "
                f"trained with {config['vocab_size']}. The files do not belong together."
            )
        return cls(
            vocab,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            weights_path=directory / "best_model.pt"
        )
    
    def predict(self, text:str) -> tuple[str, float]:
        ids = self._vocab.encode(text)
        input_tensor = torch.tensor([ids], dtype=torch.long) # adding batch dimension

        with torch.no_grad():
            logits = self._model(input_tensor)
            probs = F.softmax(logits, dim=1)

        confidence, predicted_idx = torch.max(probs, dim=1)
        label = LABELS[int(predicted_idx.item())]

        return label, confidence.item()