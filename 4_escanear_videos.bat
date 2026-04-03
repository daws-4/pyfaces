@echo off
title Py_Faces - Escaner Masivo de Videos
echo.
echo Iniciando entorno Docker para escanear videos...
echo Puede presionar Ctrl+C para detener el proceso en cualquier momento.
echo.
docker run --rm -it --memory=12g --cpus=14 -v "%~dp0:/app" py_faces_app python escaner_videos.py
echo.
pause
