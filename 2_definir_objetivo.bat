@echo off
title Py_Faces - Definir Objetivo
echo.
echo Iniciando entorno Docker para definir el perfil objetivo...
echo.
docker run --rm -it -v "%~dp0:/app" py_faces_app python definir_objetivo.py
echo.
pause
