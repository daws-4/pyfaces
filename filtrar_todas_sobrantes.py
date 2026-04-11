import os
import sys
import shutil
import pickle
from collections import Counter

# =============================================================================
# SCRIPT DE UTILIDAD: FILTRAR TODAS LAS FOTOS SIN ASIGNAR
# =============================================================================
# Variante del filtrador de sobrantes que extrae TODAS las fotos que no han
# sido asignadas a ninguna persona, sin importar cuantas caras tengan.
#
# Incluye fotos con 0, 1, 2 o mas caras detectadas.
#
# Esto es util para revisiones generales donde quieres ver TODO lo que
# quedo sin clasificar, incluyendo fotos grupales y fotos sin rostros.
# =============================================================================

# Configuración
CARPETA_ORIGEN = "fotos_prueba"
CARPETA_RESULTADOS = "output_buscador_objetivo"
CARPETA_SOBRANTES = "fotos_sobrantes_todas"
ARCHIVO_LOTE = os.path.join("output_escaner_encodings", "lote_encodings.pkl")

def buscar_todas_sobrantes():
    print("=" * 65)
    print("  FILTRADOR DE TODAS LAS FOTOS SOBRANTES (SIN FILTRO DE CARAS)")
    print("=" * 65)

    if not os.path.exists(CARPETA_ORIGEN):
        print(f"\n  Error: La carpeta de origen '{CARPETA_ORIGEN}' no existe.")
        sys.exit(1)

    # Leer datos del lote para mostrar info de caras (opcional pero informativo)
    conteo_caras_por_foto = Counter()
    if os.path.exists(ARCHIVO_LOTE):
        print(f"\n  Consultando memoria de caras ({ARCHIVO_LOTE})...", end=" ", flush=True)
        with open(ARCHIVO_LOTE, "rb") as f:
            datos_lote = pickle.load(f)
        for cara in datos_lote["caras"]:
            conteo_caras_por_foto[cara["nombre_archivo"]] += 1
        print(f"OK ({len(datos_lote['caras'])} rostros en memoria)")
    else:
        print(f"\n  Aviso: No se encontro '{ARCHIVO_LOTE}'. No se mostrara info de caras.")

    # Recopilar todas las fotos ya asignadas en resultados
    if not os.path.exists(CARPETA_RESULTADOS):
        print(f"  Aviso: La carpeta de resultados '{CARPETA_RESULTADOS}' no existe aun.")
        asignadas = set()
    else:
        asignadas = set()
        for raiz, directorios, archivos in os.walk(CARPETA_RESULTADOS):
            # Ignorar la carpeta de sobrantes si esta dentro de resultados
            if "sobrantes" in raiz.lower():
                continue
            for a in archivos:
                if not a.endswith('.xlsx'):
                    asignadas.add(a)

    print(f"  Analizando bandeja de resultados... Encontradas {len(asignadas)} copias despachadas.")

    # Encontrar todas las fotos en la carpeta de origen
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    fotos_origen = []
    for raiz, _, archivos in os.walk(CARPETA_ORIGEN):
        for nombre in archivos:
            if os.path.splitext(nombre)[1].lower() in extensiones:
                fotos_origen.append(os.path.join(raiz, nombre))

    print(f"  Analizando bandeja de entrada... Encontradas {len(fotos_origen)} fotos originales.\n")

    # Filtrar: solo las que NO han sido asignadas (sin filtro de caras)
    fotos_sin_asignar = []

    for ruta_orig in fotos_origen:
        nombre = os.path.basename(ruta_orig)

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
            fotos_sin_asignar.append(ruta_orig)

    # Clasificar por cantidad de caras para el resumen
    sin_cara = []
    una_cara = []
    multiples_caras = []

    for ruta in fotos_sin_asignar:
        nombre = os.path.basename(ruta)
        caras = conteo_caras_por_foto.get(nombre, 0)
        if caras == 0:
            sin_cara.append(ruta)
        elif caras == 1:
            una_cara.append(ruta)
        else:
            multiples_caras.append((ruta, caras))

    # Resultados
    os.makedirs(CARPETA_SOBRANTES, exist_ok=True)

    print(f"  >> FOTOS SOBRANTES TOTALES: {len(fotos_sin_asignar)}")
    print(f"     - Sin cara detectada:      {len(sin_cara)}")
    print(f"     - Con 1 cara:              {len(una_cara)}")
    print(f"     - Con multiples caras:     {len(multiples_caras)}")
    print("-" * 65)

    if len(fotos_sin_asignar) > 0:
        print(f"  Copiando a '{CARPETA_SOBRANTES}'...")
        copiadas = 0
        for ruta in fotos_sin_asignar:
            destino = os.path.join(CARPETA_SOBRANTES, os.path.basename(ruta))

            # Evitar colisiones
            if os.path.exists(destino):
                base, ext = os.path.splitext(os.path.basename(ruta))
                contador = 1
                while os.path.exists(destino):
                    destino = os.path.join(CARPETA_SOBRANTES, f"{base}_{contador}{ext}")
                    contador += 1

            try:
                shutil.copy2(ruta, destino)
                copiadas += 1
            except Exception:
                pass

        print(f"  Extraccion exitosa. Se copiaron {copiadas} fotos hacia '{CARPETA_SOBRANTES}'.")
    else:
        print("  No se detectaron fotos sobrantes. Todas han sido asignadas a alguien.")

    print(f"{'=' * 65}\n")

if __name__ == "__main__":
    buscar_todas_sobrantes()
