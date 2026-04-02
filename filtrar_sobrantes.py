import os
import sys
import shutil
import pickle
from collections import Counter

# =============================================================================
# SCRIPT DE UTILIDAD: FILTRAR FOTOS SIN ASIGNAR (1 SOLA CARA)
# =============================================================================
# Este script escanea la carpeta de entrada y la compara con la salida.
# Solo extrae como sobrantes a aquellas fotos que cumplan dos reglas de oro:
#   1. NUNCA han sido identificadas ni asignadas a ninguna carpeta de nadie.
#   2. Tienen EXACTAMENTE 1 CARA detectada por el escaner.
#
# Esto es ideal para que el resultado te entregue candidatos perfectos
# para arrastrarlos a la carpeta "persona_objetivo" y crear nuevos perfiles.
# =============================================================================

# Configuración
CARPETA_ORIGEN = "fotos_prueba"
CARPETA_RESULTADOS = "output_buscador_objetivo"
CARPETA_SOBRANTES = "fotos_sobrantes"
ARCHIVO_LOTE = os.path.join("output_escaner_encodings", "lote_encodings.pkl")

def buscar_fotos_sobrantes():
    print("=" * 65)
    print("  FILTRADOR DE FOTOS SOBRANTES (CON EXACTAMENTE 1 CARA)")
    print("=" * 65)

    if not os.path.exists(CARPETA_ORIGEN):
        print(f"\n  Error: La carpeta de origen '{CARPETA_ORIGEN}' no existe.")
        sys.exit(1)
        
    if not os.path.exists(ARCHIVO_LOTE):
        print(f"\n  Error: Falta el archivo maestro de escaneo '{ARCHIVO_LOTE}'.")
        print("  Asegurate de haber ejecutado 1_escanear_fotos.bat primero.")
        sys.exit(1)

    # 1. Leer los datos de caras escaneadas para saber cuantas caras tiene cada foto original
    print(f"\n  Consultando memoria de caras ({ARCHIVO_LOTE})...", end=" ", flush=True)
    with open(ARCHIVO_LOTE, "rb") as f:
        datos_lote = pickle.load(f)
        
    conteo_caras_por_foto = Counter()
    for cara in datos_lote["caras"]:
        conteo_caras_por_foto[cara["nombre_archivo"]] += 1
    print(f"OK ({len(datos_lote['caras'])} rostros encontrados en memoria)")

    if not os.path.exists(CARPETA_RESULTADOS):
        print(f"  Aviso: La carpeta de resultados '{CARPETA_RESULTADOS}' no existe aun.")
        asignadas = set()
    else:
        # Recopilar todos los nombres de archivo exactos que han sido copiados a resultados
        asignadas = set()
        for raiz, directorios, archivos in os.walk(CARPETA_RESULTADOS):
            if CARPETA_SOBRANTES in raiz: 
                continue
            for a in archivos:
                if not a.endswith('.xlsx'):
                    asignadas.add(a)

    print(f"  Analizando bandeja de resultados... Encontradas {len(asignadas)} copias despachadas.")

    # Encontrar las fotos en la bandeja original
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    fotos_origen = []
    for raiz, _, archivos in os.walk(CARPETA_ORIGEN):
        for nombre in archivos:
            if os.path.splitext(nombre)[1].lower() in extensiones:
                fotos_origen.append(os.path.join(raiz, nombre))

    print(f"  Analizando bandeja de entrada... Encontradas {len(fotos_origen)} fotos originales.\n")

    # Hacer el match evaluando si estan asignadas Y el filtro de 1 sola cara
    fotos_sin_asignar_una_cara = []
    
    for ruta_orig in fotos_origen:
        nombre = os.path.basename(ruta_orig)
        
        # Filtro de Inteligencia: Contar si en la base de datos tuvo 1 cara
        caras_detectadas = conteo_caras_por_foto.get(nombre, 0)
        if caras_detectadas != 1:
            continue
            
        fue_asignada = False
        
        # 1. Comprobar si existe el nombre exacto
        if nombre in asignadas:
            fue_asignada = True
        else:
            # 2. Comprobar si existe el nombre con sufijos de copia (ej. IMG_1_1.jpg)
            base, ext = os.path.splitext(nombre)
            for i in range(1, 15):
                if f"{base}_{i}{ext}" in asignadas:
                    fue_asignada = True
                    break
                    
        if not fue_asignada:
            fotos_sin_asignar_una_cara.append(ruta_orig)

    # Resultados
    os.makedirs(CARPETA_SOBRANTES, exist_ok=True)
    
    print(f"  >> FOTOS SOBRANTES Y CON 1 SOLA CARA: {len(fotos_sin_asignar_una_cara)}")
    print("-" * 65)
    
    if len(fotos_sin_asignar_una_cara) > 0:
        print(f"  Copiando a '{(CARPETA_SOBRANTES)}'...")
        copiadas = 0
        for ruta in fotos_sin_asignar_una_cara:
            destino = os.path.join(CARPETA_SOBRANTES, os.path.basename(ruta))
            
            # Evitar colisiones en destino por si hay subcarpetas de origen con el mismo nombre de foto
            if os.path.exists(destino):
                base, ext = os.path.splitext(os.path.basename(ruta))
                contador = 1
                while os.path.exists(destino):
                    destino = os.path.join(CARPETA_SOBRANTES, f"{base}_{contador}{ext}")
                    contador += 1
            
            try:
                shutil.copy2(ruta, destino)
                copiadas += 1
            except Exception as e:
                pass
                
        print(f"  Extraccion exitosa. Se copiaron {copiadas} fotos hacia la carpeta '{CARPETA_SOBRANTES}'.")
        print("  !Listas para que las uses como perfiles en el paso 2 (definir objetivo)!")
    else:
        print("  No se detectaron sobrantes perfectos (con 1 cara). Todas han sido")
        print("  asignadas o las que sobran son grupales/sin personas.")

    print(f"{'=' * 65}\n")

if __name__ == "__main__":
    buscar_fotos_sobrantes()
