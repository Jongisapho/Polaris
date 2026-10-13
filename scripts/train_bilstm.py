from sentiment_core.data import load_splits
from sentiment_core.training import TrainConfig, train

train_data, val_data, _ = load_splits()    # the test split is deliberately ignored
history = train(train_data, val_data, TrainConfig())