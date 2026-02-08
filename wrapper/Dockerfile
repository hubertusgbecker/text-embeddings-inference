FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir flask gunicorn requests

COPY wrapper.py .

EXPOSE 80

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:80", "--timeout", "30", "wrapper:app"]
