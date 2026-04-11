# Usa una imagen oficial de Python
# Python 3.11 es maduro y estable para face_recognition
FROM python:3.11-slim

# Instalar dependencias del sistema requeridas para compilar dlib (C++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las librerías de Python. 
# Esto compilará dlib, lo cual puede tomar unos minutos la primera vez.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# (Nota: El código fuente se montará como volumen desde el host al ejecutar,
#  por lo tanto no copiamos los scripts locales a la imagen aquí)
