@echo off
title Py_Faces - Construir Imagen Docker
echo.
echo ====================================================================
echo   Py_Faces: Construyendo Imagen Docker
echo ====================================================================
echo.
echo Esta accion descargara el entorno Linux virtual y compilara el
echo codigo pesado de C++ en un contenedor aislado.
echo.
echo IMPORTANTE: Puede tardar unos 10 a 15 minutos la primera vez.
echo Por favor, se paciente y no cierres la ventana.
echo.
docker build -t py_faces_app .
echo.
echo ====================================================================
echo   CONSTRUCCION FINALIZADA
echo ====================================================================
echo Si no viste ningun mensaje de error en rojo al final, todo
echo se ha instalado correctamente. 
echo Ya puedes ejecutar los otros scripts (pasos 1, 2, 3...)
echo.
pause
