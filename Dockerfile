FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    NLTK_DATA=/usr/share/nltk_data \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install locked dependencies first so this layer is cached until pyproject/uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# NLTK data for the BM25 tokenizer. Safeguard: never fails the build if it is not needed.
RUN python -c "import nltk; [nltk.download(p, download_dir='/usr/share/nltk_data', quiet=True) for p in ('punkt', 'punkt_tab', 'stopwords')]" || true

# App code (see .dockerignore: .venv, .env, eval/, dashboard.py are left out)
COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 8000
# Render injects $PORT; falls back to 8000 locally
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]