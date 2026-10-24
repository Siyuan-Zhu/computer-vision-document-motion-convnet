# Batch prediction on CPU: classify a folder of images and write a CSV.
#
#   docker build -t doc-motion .
#   docker run --rm \
#     -v "$PWD/models:/models:ro" -v "$PWD/frames:/frames:ro" -v "$PWD/out:/out" \
#     doc-motion --model /models/model.pt --images /frames --out /out/predictions.csv
#
# Both model formats work: .pt (PyTorch port) and .keras (Keras on the
# PyTorch backend).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KERAS_BACKEND=torch

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

RUN useradd --create-home predictor
USER predictor

ENTRYPOINT ["doc-motion", "predict"]
