from abc import ABC, abstractmethod # abstractclass in python
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
            "negatiev" or "neutral" and confidence is a float in 
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