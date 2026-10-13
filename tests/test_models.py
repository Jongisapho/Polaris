import pytest
from sentiment_core.models import VaderSentimentModel, BiLSTMSentimentModel
from sentiment_core.preprocessing import Vocabulary

@pytest.fixture
def vocab() -> Vocabulary:
    v = Vocabulary()
    v.build([
        "This project is great",
        "Terrible debugging User interface",
        "It was okay, I guess ..",
        "I absolutely loved it",
        "Worst experience ever"
    ])
    return v

@pytest.fixture
def vader_model() -> VaderSentimentModel:
    return VaderSentimentModel()

@pytest.fixture
def bilstm_model(vocab: Vocabulary) -> BiLSTMSentimentModel:
    return BiLSTMSentimentModel(vocab=vocab)

@pytest.mark.parametrize("model_fixture", ["vader_model", "bilstm_model"])
def test_predict_returns_valid_tuple(model_fixture: str, request:pytest.FixtureRequest) -> None:
    model = request.getfixturevalue(model_fixture)
    label, confidence = model.predict("This is a normal test sentence.")
    assert isinstance(label, str)
    assert label in ("positive", "negative", "neutral")
    assert isinstance(confidence, float)

@pytest.mark.parametrize("model_fixture", ["vader_model", "bilstm_model"])
def test_predict_batch_returns_list_of_tuples(model_fixture: str, request: pytest.FixtureRequest) -> None:
    model = request.getfixturevalue(model_fixture)

    texts = ["Great product presentation",
             "I Love this experience",
             "Worst feeling though good services"]
    results = model.predict_batch(texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)
    for label, confidence in results:
        assert label in ("positive", "negative", "neutral")
        assert isinstance(confidence, float)

@pytest.mark.parametrize("model_fixture", ["vader_model", "bilstm_model"])
def test_predict_handles_empty_string(model_fixture: str, request: pytest.FixtureRequest) -> None:
    model = request.getfixturevalue(model_fixture)

    label, confidence = model.predict("")
    assert label in ("positive", "negative", "neutral")
    assert isinstance(confidence, float)
