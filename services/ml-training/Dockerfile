FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/_common ./_common
COPY services/ml-training/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir ./_common

COPY services/ml-training/main.py ./
COPY services/ml-training/pipeline.py ./

EXPOSE 9121

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:9121/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9121"]
