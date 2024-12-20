FROM python:3.9-slim

LABEL maintainer="yunuscancengizz@gmail.com"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]