import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    """
    BiLSTM-based sentiment classifier
    Embedding -> Bidirectional LSTM -> Linear -> logits over 3 classes
    """

    def __init__(
        self,
        vocab_size:int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        _, (hidden, _) = self.lstm(embedded) # hidden: (2, batch, hidden_dim)

        forward_hidden = hidden[0] # last forward-direction hidden state
        backward_hidden = hidden[1] # last backward-direction hidden state
        combined = torch.cat((forward_hidden, backward_hidden), dim=1) # (batch, hidden_dim * 2)

        logits = self.fc(combined) # (batch, num_classes)
        return logits