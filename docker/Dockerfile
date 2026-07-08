FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY ocrbench ./ocrbench
COPY apps ./apps

EXPOSE 8000 7860

CMD ["python", "-m", "ocrbench.benchmark.pipeline"]
