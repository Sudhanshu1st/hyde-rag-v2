# Thin wrappers around the cross-platform CLI (`uv run hyde-rag ...`).
# Windows without make: run the commands after each target directly.

.PHONY: setup lint test reproduce-beir eval-food demo api quota

setup:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

# Re-scores every BEIR system from the committed LLM cache: no API key needed.
# TREC-COVID dense encoding is ~30 CPU-hours; use notebooks/encode_trec_covid_colab.ipynb on a
# free GPU and `hyde-rag verify-index` instead.
reproduce-beir:
	uv run python scripts/download_beir.py scifact trec-covid
	uv run python scripts/download_scifact_labels.py
	uv run hyde-rag encode --dataset scifact --encoders contriever bge-base
	uv run hyde-rag run configs/experiments/scifact.yaml
	uv run hyde-rag run configs/experiments/scifact-dev.yaml
	uv run hyde-rag gen-eval configs/experiments/scifact-generation.yaml --no-llm
	uv run hyde-rag run configs/experiments/trec-covid.yaml

eval-food:
	uv run python scripts/download_food_corpus.py
	uv run hyde-rag ingest-food
	uv run hyde-rag index-food --encoders bge-base

api:
	uv run uvicorn hyde_rag.api.app:app --port 8000

demo:
	uv run streamlit run app/streamlit_app.py

quota:
	uv run hyde-rag quota
