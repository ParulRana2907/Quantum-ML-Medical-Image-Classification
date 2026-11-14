
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip


RUN python -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --retries 10 \
    --timeout 120 \
    --prefer-binary \
    -r requirements.txt


COPY app.py .
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
