@echo off
title Py_Faces - Buscador de Objetivo
echo.
echo Iniciando entorno Docker para buscar coincidencias...
echo.
docker run --rm -it -v "%~dp0:/app" py_faces_app python buscador_objetivo.py
echo.
pause
