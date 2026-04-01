FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para OpenCV, PyTorch y Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requisitos e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar el modelo (puedes también copiarlo localmente si prefieres)
# Nota: el modelo pesa varios cientos de MB, se descarga en tiempo de build
RUN wget -O yoloe-26x-seg.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt

# Copiar el handler
COPY handler.py .

# Exponer puerto para FastAPI
EXPOSE 8000

# Comando para iniciar el servidor
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000"]
