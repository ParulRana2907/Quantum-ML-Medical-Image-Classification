
FROM python:3.11-slim

WORKDIR /app

COPY requirements-serve.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install --no-cache-dir -r requirements-serve.txt

COPY app.py .
COPY model/ ./model/

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
