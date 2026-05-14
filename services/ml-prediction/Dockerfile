FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/_common ./_common
COPY services/ml-prediction/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir ./_common

COPY services/ml-prediction/main.py ./

EXPOSE 9122

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:9122/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9122"]
