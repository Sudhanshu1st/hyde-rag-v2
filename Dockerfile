# API image (CPU-only). Corpus, indexes and model caches are mounted, not baked in:
#   docker build -t hyde-rag-v2 .
#   docker run -p 8000:8000 --env-file .env \
#     -v "$PWD/data:/app/data" -v "$PWD/cache:/app/cache" -v hf-cache:/root/.cache/huggingface \
#     hyde-rag-v2
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN pip install --no-cache-dir uv==0.12.10

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN uv sync --frozen --no-dev --extra api --no-install-project
COPY src ./src
COPY configs ./configs
COPY data_sources.yaml ./
RUN uv sync --frozen --no-dev --extra api

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uv", "run", "--no-sync", "uvicorn", "hyde_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
