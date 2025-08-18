FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -v -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn appmain:app --host 0.0.0.0 --port 8000 2>&1 | grep -v 'tensorflow' || true"]
