# Polaris

Comparing a rule-based sentiment model (VADER) against a deep learning model (BiLSTM) as part of a research, built into a distributed pipeline: Python (FastAPI) for model inference, Java (Spring Boot) for ingestion and orchestration, PostgreSQL for result storage.

## Status
In Development - Phase 1: Python ML Core

## Structure
- `sentiment_core/` - model implementations (VADER, BiLSTM) behind a shared interface
- `tests/` - unit tests
- `data/` - dataset (not committed)