import hashlib
import sys
import os
import time

# Hash SHA-256 de la contraseña esperada.
# Valor por defecto generado para la contraseña: "codePicture/14"
HASH_ESPERADO = "c4e3d44a952a51405b42eef5459eeadbe8d05dcf0d536775f9948e86001f166b"

def validar_acceso():
    archivo_token = ".auth_ok"
    
    # Si el archivo token existe, verificamos su antigüedad
    if os.path.exists(archivo_token):
        # Si tiene menos de 12 horas, permite el acceso directo
        # (Así el usuario no tiene que poner la clave en cada script)
        tiempo_modificacion = os.path.getmtime(archivo_token)
        if (time.time() - tiempo_modificacion) < (12 * 3600):
            return True
        else:
            # Si caducó, lo borramos y pedimos de nuevo
            try:
                os.remove(archivo_token)
            except:
                pass
            
    print("\n=======================================================")
    print("   ACCESO RESTRINGIDO - LICENCIA DE PY_FACES")
    print("=======================================================")
    
    try:
        clave = input("Ingrese el código de acceso / licencia: ").strip()
    except Exception:
        print("Error al leer la entrada del teclado.")
        sys.exit(1)
        
    # Calculamos el hash de lo que ingresó el usuario
    hash_calculado = hashlib.sha256(clave.encode('utf-8')).hexdigest()
    
    if hash_calculado == HASH_ESPERADO:
        print(">> Acceso concedido.\n")
        # Guardamos un token temporal para los siguientes scripts
        try:
            with open(archivo_token, "w") as f:
                f.write("valido")
        except:
            pass
        return True
    else:
        print(">> Acceso denegado. Código incorrecto.")
        sys.exit(1)

# Ejecutar validación inmediata al importar el módulo
validar_acceso()
