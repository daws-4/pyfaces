@echo off
title Py_Faces - Buscar Nueva Persona
echo.
echo ====================================================================
echo   FLUJO AUTOMATICO: DEFINIR OBJETIVO + BUSCAR MASIVAMENTE
echo ====================================================================
echo.
echo Este proceso ejecutara los scripts 2 y 3 de forma conjunta para 
echo ahorrarte tiempo. 
echo.

docker run --rm -it -v "%~dp0:/app" py_faces_app python definir_objetivo.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Hubo un error al procesar el rostro.
    echo     La busqueda masiva ha sido cancelada.
    echo.
    pause
    exit /b
)

echo.
echo --------------------------------------------------------------------
echo Perfil creado con exito. Ejecutando la busqueda instantanea...
echo --------------------------------------------------------------------
echo.

docker run --rm -it -v "%~dp0:/app" py_faces_app python buscador_objetivo.py

echo.
echo ====================================================================
echo   PROCESO INTEGRAL FINALIZADO
echo ====================================================================
echo Ya puedes borrar la vieja foto de la carpeta 'persona_objetivo',
echo arrastrar la foto de tu siguiente victima/cliente y volver
echo a dar doble clic a este mismo boton automatico.
echo.
pause
