# Polaris

** Distributed Sentiment Analysis Pipeline

Polaris turns a comparative sentiment analysis study into a fully working, scalable system.
A Java Spring Boot gateway handles ingestion, rate-limiting and queuing.
A Python FastAPI service runs the models.
PostgreSQL is the single source of truth.
Kafka (Optional) decouples the two sides so the system remains resilient under load.
