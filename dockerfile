FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer separately
# from the code — much faster rebuilds when only source files change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Train the model at build time so the image is self-contained and
# doesn't need a separate training step before it can serve requests.
# For a real production system you'd instead pull a specific model
# version from an artifact store (e.g. MLflow Model Registry, S3) —
# see the README for that extension.
RUN python -m src.data && python -m src.train

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]