import pytest
from sentiment_core.data import load_splits
from sentiment_core.labels import LABELS

SPLIT_NAMES = ["train", "validation", "test"]
EXPECTED_SIZES = {"train": 45615, "validation": 2000, "test": 12284}

@pytest.fixture(scope="module")
def splits():
    return load_splits()

def class_share(labels: list[int], cls: int) -> float:
    return labels.count(cls) / len(labels)

@pytest.mark.parametrize("idx, name", list(enumerate(SPLIT_NAMES)))
def test_split_is_non_empty_and_aligned(splits, idx, name):
    texts, labels = splits[idx]
    assert len(texts) > 0 , f"{name} split is empty"
    assert len(texts) == len(labels), f"{name}: texts and labels differ in length"
    assert all(isinstance(t, str) for t in texts)

@pytest.mark.parametrize("idx, name", list(enumerate(SPLIT_NAMES)))
def test_labels_are_valid(splits, idx, name):
    _, labels = splits[idx]
    assert set(labels) == set(range(len(LABELS))), f"{name}: unexpected label set"

@pytest.mark.parametrize("idx, name", list(enumerate(SPLIT_NAMES)))
def test_split_sizes_are_pinned(splits, idx, name):
    texts, _ = splits[idx]
    assert len(texts) == EXPECTED_SIZES[name]

def test_validation_resembles_train(splits):
    train, validation, _ = splits
    for cls in range(len(LABELS)):
        gap = abs(class_share(train[1], cls) - class_share(validation[1], cls))
        assert gap < 0.03, f"{LABELS[cls]}: train/validation gap is {gap:.3f}"


def test_test_split_is_shifted(splits):
    train, _, test = splits
    neg = LABELS.index("negative")
    pos = LABELS.index("positive")

    neg_gain = class_share(test[1], neg) - class_share(train[1], neg)
    pos_drop = class_share(train[1], pos) - class_share(test[1], pos)

    assert neg_gain > 0.10, "expected many more negatives in test than train"
    assert pos_drop > 0.10, "expected many fewer positives in test than train"