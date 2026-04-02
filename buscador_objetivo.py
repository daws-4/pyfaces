import os
import sys
import time
import pickle
import shutil
import numpy as np
import pandas as pd
import face_recognition

# =============================================================================
# PASO 3 DE 3: BUSCAR PERSONA EN EL LOTE DE IMAGENES
# =============================================================================
# Este script carga:
#   1. El lote de encodings generado por escaner_encodings.py
#   2. El perfil de persona generado por definir_objetivo.py
#
# Compara cada cara del lote contra el perfil objetivo y genera un Excel
# con todas las fotos donde aparece esa persona.
#
# Este paso es INSTANTANEO (segundos) porque solo compara numeros.
# Puedes ejecutarlo multiples veces con diferentes perfiles o tolerancias.
# =============================================================================

# --- CONFIGURACION ---
ARCHIVO_LOTE = os.path.join("output_escaner_encodings", "lote_encodings.pkl")
ARCHIVO_PERFIL = os.path.join("output_comparador_caras", "perfil_objetivo.pkl")
CARPETA_SALIDA = "output_buscador_objetivo"
ARCHIVO_EXCEL = os.path.join(CARPETA_SALIDA, "resultado_busqueda.xlsx")

# Tolerancia de coincidencia (ajustable sin re-escanear)
# 0.40 = muy estricto (pocas coincidencias, alta precision)
# 0.45 = equilibrado
# 0.50 = mas flexible (mas coincidencias, puede incluir falsos positivos)
TOLERANCIA_BUSQUEDA = 0.45


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PASO 3/3: BUSCAR PERSONA EN LOTE DE IMAGENES")
    print("=" * 65)

    # --- Cargar lote de encodings ---
    if not os.path.isfile(ARCHIVO_LOTE):
        print(f"\n  No se encontro '{ARCHIVO_LOTE}'.")
        print(f"  Primero ejecuta: python escaner_encodings.py")
        sys.exit(1)

    print(f"\n  Cargando lote: {ARCHIVO_LOTE}...", end=" ", flush=True)
    with open(ARCHIVO_LOTE, "rb") as f:
        datos_lote = pickle.load(f)
    caras_lote = datos_lote["caras"]
    metadata_lote = datos_lote["metadata"]
    print(f"OK ({len(caras_lote)} caras de {metadata_lote['total_imagenes']} imagenes)")

    # --- Cargar perfil objetivo ---
    if not os.path.isfile(ARCHIVO_PERFIL):
        print(f"\n  No se encontro '{ARCHIVO_PERFIL}'.")
        print(f"  Primero ejecuta: python definir_objetivo.py")
        sys.exit(1)

    print(f"  Cargando perfil: {ARCHIVO_PERFIL}...", end=" ", flush=True)
    with open(ARCHIVO_PERFIL, "rb") as f:
        perfil = pickle.load(f)

    nombre_persona = perfil["nombre"]
    encoding_objetivo = np.array(perfil["encoding_promedio"])
    todos_enc_persona = [np.array(e) for e in perfil["todos_encodings"]]
    num_muestras = perfil["num_muestras"]
    print(f"OK ({nombre_persona}, {num_muestras} muestras)")

    print(f"\n  Tolerancia de busqueda: {TOLERANCIA_BUSQUEDA}")
    print(f"  Origen del lote: {metadata_lote.get('carpeta_origen', 'desconocido')}")
    print("-" * 65)

    # --- Comparacion ---
    print(f"\n  Comparando {len(caras_lote)} caras contra '{nombre_persona}'...\n")
    tiempo_inicio = time.time()

    # Convertir todos los encodings del lote a una matriz numpy
    encodings_lote = np.array([cara["encoding"] for cara in caras_lote])

    # Calcular distancia de cada cara del lote contra el encoding promedio
    distancias_promedio = face_recognition.face_distance(encodings_lote, encoding_objetivo)

    # Para mayor precision: verificar contra TODAS las muestras del perfil
    # Una cara es coincidencia si la distancia promedio a todas las muestras <= tolerancia
    coincidencias = []
    fotos_con_coincidencia = set()

    for i, cara in enumerate(caras_lote):
        dist_al_promedio = distancias_promedio[i]

        # Filtro rapido: si la distancia al promedio es muy alta, saltarla
        if dist_al_promedio > TOLERANCIA_BUSQUEDA + 0.1:
            continue

        # Verificacion exhaustiva: comparar contra todas las muestras
        if len(todos_enc_persona) > 1:
            distancias_a_muestras = face_recognition.face_distance(
                todos_enc_persona, np.array(cara["encoding"])
            )
            dist_media = np.mean(distancias_a_muestras)
            dist_minima = np.min(distancias_a_muestras)
        else:
            dist_media = dist_al_promedio
            dist_minima = dist_al_promedio

        # Es coincidencia si la distancia media a las muestras esta dentro de tolerancia
        if dist_media <= TOLERANCIA_BUSQUEDA:
            # Calcular nivel de confianza
            if dist_media <= 0.35:
                confianza = "MUY ALTA"
            elif dist_media <= 0.40:
                confianza = "ALTA"
            elif dist_media <= 0.45:
                confianza = "MEDIA"
            else:
                confianza = "BAJA"

            coincidencias.append({
                "Archivo": cara["nombre_archivo"],
                "Ruta Completa": cara["ruta_archivo"],
                "Distancia Promedio": round(dist_media, 4),
                "Distancia Minima": round(dist_minima, 4),
                "Confianza": confianza
            })
            fotos_con_coincidencia.add(cara["ruta_archivo"])

    tiempo_busqueda = time.time() - tiempo_inicio

    # --- Mostrar resultados ---
    print(f"  Busqueda completada en {tiempo_busqueda:.2f}s")
    print(f"\n  RESULTADOS PARA: {nombre_persona}")
    print(f"  {'=' * 50}")
    print(f"  Coincidencias encontradas: {len(coincidencias)}")
    print(f"  Fotos unicas con esta persona: {len(fotos_con_coincidencia)}")

    if len(coincidencias) > 0:
        # Ordenar por distancia (mejor coincidencia primero)
        coincidencias.sort(key=lambda x: x["Distancia Promedio"])

        # Estadisticas
        distancias = [c["Distancia Promedio"] for c in coincidencias]
        print(f"  Distancia promedio: {np.mean(distancias):.4f}")
        print(f"  Mejor coincidencia: {min(distancias):.4f}")
        print(f"  Peor coincidencia:  {max(distancias):.4f}")

        # Mostrar las mejores coincidencias en consola
        print(f"\n  Top 10 mejores coincidencias:")
        for i, c in enumerate(coincidencias[:10], 1):
            print(f"    {i}. [{c['Confianza']}] {c['Archivo']} (dist: {c['Distancia Promedio']})")

        # --- Exportar a Excel ---
        # Hoja 1: Coincidencias detalladas
        df_coincidencias = pd.DataFrame(coincidencias)

        # Hoja 2: Resumen por foto unica (sin duplicados si hay multiples caras)
        fotos_unicas = {}
        for c in coincidencias:
            ruta = c["Ruta Completa"]
            if ruta not in fotos_unicas or c["Distancia Promedio"] < fotos_unicas[ruta]["Mejor Distancia"]:
                fotos_unicas[ruta] = {
                    "Archivo": c["Archivo"],
                    "Ruta Completa": ruta,
                    "Mejor Distancia": c["Distancia Promedio"],
                    "Confianza": c["Confianza"]
                }

        fotos_resumen = sorted(fotos_unicas.values(), key=lambda x: x["Mejor Distancia"])
        df_resumen = pd.DataFrame(fotos_resumen)

        # Hoja 3: Todas las fotos del lote con indicador si/no
        todas_fotos = {}
        for cara in caras_lote:
            ruta = cara["ruta_archivo"]
            if ruta not in todas_fotos:
                todas_fotos[ruta] = {
                    "Archivo": cara["nombre_archivo"],
                    "Ruta Completa": ruta,
                    f"Aparece {nombre_persona}": "NO"
                }
            if ruta in fotos_con_coincidencia:
                todas_fotos[ruta][f"Aparece {nombre_persona}"] = "SI"

        df_todas = pd.DataFrame(sorted(todas_fotos.values(), key=lambda x: x["Archivo"]))

        # Guardar Excel con 3 hojas
        os.makedirs(CARPETA_SALIDA, exist_ok=True)
        with pd.ExcelWriter(ARCHIVO_EXCEL, engine='openpyxl') as writer:
            df_coincidencias.to_excel(writer, sheet_name="Coincidencias", index=False)
            df_resumen.to_excel(writer, sheet_name="Fotos Unicas", index=False)
            df_todas.to_excel(writer, sheet_name="Lote Completo", index=False)

        print(f"\n  Excel generado: {ARCHIVO_EXCEL}")
        print(f"     -> Hoja 'Coincidencias' ({len(coincidencias)} filas)")
        print(f"     -> Hoja 'Fotos Unicas' ({len(fotos_resumen)} fotos)")
        print(f"     -> Hoja 'Lote Completo' ({len(todas_fotos)} fotos totales)")

        # --- Copiar imagenes donde aparece el objetivo ---
        carpeta_objetivo = os.path.join(CARPETA_SALIDA, nombre_persona)
        os.makedirs(carpeta_objetivo, exist_ok=True)

        copiadas = 0
        errores_copia = 0
        for ruta in fotos_con_coincidencia:
            # Reemplazar barras cruzadas por si el archivo .pkl fue creado en Windows 
            # originalmente y ahora está siendo leído dentro del contenedor Linux.
            ruta_arreglada = ruta.replace('\\', os.sep).replace('/', os.sep)
            
            # Autocorreccion avanzada: Si la ruta tiene pegado "C:\Users\..." (Ruta absoluta de Windows)
            # Docker no lo encontrara porque el contenedor empieza desde "/app". 
            # Le quitaremos esa "basura" quedandonos solo de la carpeta de fotos hacia adelante.
            if not os.path.isfile(ruta_arreglada) and 'fotos_prueba' in ruta_arreglada:
                partes = ruta_arreglada.split('fotos_prueba')
                # Reconstruir la ruta local
                ruta_arreglada = os.path.join('fotos_prueba', partes[-1].lstrip(os.sep))

            if os.path.isfile(ruta_arreglada):
                nombre_destino = os.path.basename(ruta_arreglada)
                destino = os.path.join(carpeta_objetivo, nombre_destino)
                # Si ya existe un archivo con el mismo nombre, agregar sufijo
                if os.path.exists(destino):
                    base, ext = os.path.splitext(nombre_destino)
                    contador = 1
                    while os.path.exists(destino):
                        destino = os.path.join(carpeta_objetivo, f"{base}_{contador}{ext}")
                        contador += 1
                try:
                    shutil.copy2(ruta_arreglada, destino)
                    copiadas += 1
                except Exception:
                    errores_copia += 1
            else:
                errores_copia += 1

        print(f"\n  Imagenes copiadas: {copiadas} en '{carpeta_objetivo}'")
        if errores_copia > 0:
            print(f"  {errores_copia} imagenes no se pudieron copiar (archivo no encontrado)")

    else:
        print(f"\n  No se encontraron coincidencias con tolerancia {TOLERANCIA_BUSQUEDA}.")
        print(f"  Sugerencias:")
        print(f"    - Aumentar TOLERANCIA_BUSQUEDA (ej: 0.50)")
        print(f"    - Agregar mas fotos de referencia en definir_objetivo.py")
        print(f"    - Verificar que las fotos del perfil son claras y frontales")

    print(f"\n{'=' * 65}")
    print(f"  BUSQUEDA FINALIZADA")
    print(f"  Persona: {nombre_persona}")
    print(f"  Lote: {metadata_lote['total_imagenes']} imagenes ({len(caras_lote)} caras)")
    print(f"  Resultado: {len(fotos_con_coincidencia)} fotos con coincidencia")
    print(f"{'=' * 65}")
