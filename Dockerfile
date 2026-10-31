FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY artifacts ./artifacts
COPY scripts ./scripts
COPY src ./src

EXPOSE 8000

CMD ["python", "scripts/server.py"]