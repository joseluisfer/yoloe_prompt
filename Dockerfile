FROM python:3.10-slim-bookworm

WORKDIR /app

# No usar apt-get, instalar solo Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelo
RUN wget -O yoloe-26x-seg.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt

COPY handler.py .

EXPOSE 8000

CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000"]
