@echo off
title Py_Faces - Filtrar Sobrantes
echo.
echo Iniciando entorno Docker para auditar fotografias sobrantes...
echo.
docker run --rm -it -v "%~dp0:/app" py_faces_app python filtrar_sobrantes.py
echo.
pause
