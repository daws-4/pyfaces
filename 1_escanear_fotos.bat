@echo off
title Py_Faces - Escaner Masivo de Fotos
echo.
echo Iniciando entorno Docker para escanear encodings...
echo Puede presionar Ctrl+C para detener el proceso en cualquier momento.
echo.
docker run --rm -it -v "%~dp0:/app" py_faces_app python escaner_encodings.py
echo.
pause
