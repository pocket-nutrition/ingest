FROM python:3.15-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/

RUN useradd -r -u 1001 appuser
USER appuser

CMD ["python", "-m", "pipeline.main"]
