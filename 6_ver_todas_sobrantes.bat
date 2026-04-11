@echo off
title Py_Faces - Filtrar Todas las Sobrantes
echo.
echo Iniciando entorno Docker para extraer TODAS las fotos sin asignar...
echo.
docker run --rm -it -v "%~dp0:/app" py_faces_app python filtrar_todas_sobrantes.py
echo.
pause
