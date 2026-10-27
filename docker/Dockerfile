FROM python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY docker/worker.py /app/worker.py
COPY tabular_lab/__init__.py tabular_lab/config.py tabular_lab/models.py tabular_lab/early_stopping.py tabular_lab/runtime.py /app/tabular_lab/
USER 65534:65534
ENTRYPOINT ["python", "/app/worker.py"]
