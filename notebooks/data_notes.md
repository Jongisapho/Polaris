# Data Notes: TweetEval Sentiment

## Dataset
- Source: `cardiffnlp/tweet_eval`, config `sentiment` (Hugging Face)
- Labels: 0 = negative, 1 = neutral, 2 = positive
- Pre-defined splits, used as provided

| Split      | Rows   |
|------------|--------|
| Train      | 45,615 |
| Validation | 2,000  |
| Test       | 12,284 |

## Class distribution

| Split      | Negative | Neutral | Positive |
|------------|----------|---------|----------|
| Train      | 15.5%    | 45.3%   | 39.1%    |
| Validation | 15.6%    | 43.5%   | 40.9%    |
| Test       | 32.3%    | 48.3%   | 19.3%    |

**Observations**
- Train and validation have similar distributions; the test set does not. Negative roughly doubles and positive roughly halves.
- Validation scores therefore cannot reveal how well a model copes with this shift. Both validation and test results will be reported.
- Neutral is the largest class in every split. An always-neutral predictor scores 48.3% accuracy on test, but only about 0.22 macro-F1. Accuracy alone is misleading here, so macro-F1 and per-class F1 are the primary metrics.

## Text length (whitespace tokens, train)

| Median | 90th pct | 95th pct | 99th pct | Max |
|--------|----------|----------|----------|-----|
| 20     | 25       | 27       | 29       | 35  |

The previous default `max_len=40` pads every tweet unnecessarily.

## Vocabulary (lowercased whitespace tokens, train only)
- Distinct tokens: 91,632
- Tokens appearing once: about 61,300 (roughly 67% of the vocabulary), a very long noisy tail

| min_freq | Vocabulary size |
|----------|-----------------|
| 1        | 91,632          |
| 2        | 30,305          |
| 3        | 19,568          |
| 5        | 12,331          |

Most frequent tokens are function words (`the`, `to`, `in`, `a`) and the `@user` placeholder.

## Tokenization observations
- Punctuation stays attached to words ("Hogwarts.", "Thursday,", "night!!!"), so variants of the same word get separate entries.
- Quote characters are attached to the first and last tokens of some tweets.
- Hashtags are mostly unique (e.g. `#HappyBirthdayRemusLupin`) and fall below `min_freq`, becoming `<unk>` even when they carry sentiment.
- `may` and `tomorrow` are unusually frequent (6,408 and 4,497 occurrences), which may point to time-bound or event-driven data collection. To check with a sample of tweets.
- Neutral tweets are often factual reporting (sports lineups, news), unlike opinion-style positive/negative tweets.
- These issues affect the BiLSTM only. VADER works on raw text, including capitalization and punctuation, so it receives unmodified text.

## Decisions (initial; revisit using validation results)

| Decision     | Value                            | Evidence                                           |
|--------------|----------------------------------|----------------------------------------------------|
| `max_len`    | 30                               | Covers over 99% of tweets (99th pct = 29)          |
| `min_freq`   | 3                                | About 19.6k words; limits embedding size and noise |
| Metrics      | Macro-F1 plus per-class F1       | Class imbalance and train/test shift               |
| Class weights| To be tested in training         | Negative is a minority in train (15.5%)            |