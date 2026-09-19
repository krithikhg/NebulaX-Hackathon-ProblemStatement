# TrainWhisper on Cloud Run: one process serves the web app and runs the same
# Python pipelines that write the submission CSVs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py door.py acv.py rail.py rail_io.py rail_features.py shm.py server.py ./
COPY models ./models
COPY public ./public

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
