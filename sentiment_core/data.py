from datasets import load_dataset

Split = tuple[list[str], list[int]]

def load_splits() -> tuple[Split, Split, Split]:
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train_texts, train_labels = ds["train"]["text"], ds["train"]["label"]
    val_texts, val_labels = ds["validation"]["text"], ds["validation"]["label"]
    test_texts, test_labels = ds["test"]["text"], ds["test"]["label"]
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)