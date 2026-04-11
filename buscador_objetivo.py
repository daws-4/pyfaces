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
#
# SISTEMA ANTI-FALSOS POSITIVOS (v2):
#   El matching usa MULTIPLES CAPAS de filtrado para eliminar falsos
#   positivos, especialmente critico para caras de ninos donde dlib
#   produce encodings muy similares entre individuos diferentes.
#
#   Capas de filtrado:
#     1. Tolerancia de distancia (media de distancias a todas las muestras)
#     2. Votacion por consenso (% de muestras que reconocen la cara)
#     3. Filtro de consistencia (desviacion estandar de distancias baja)
#     4. Verificacion por mediana (mediana <= tolerancia)
#     5. Score compuesto de confianza (combinacion ponderada de metricas)
#
#   Las coincidencias que pasan ALGUNAS pero NO TODAS las capas van a una
#   hoja separada "Revision Manual" en el Excel para inspeccion humana.
# =============================================================================

# --- CONFIGURACION ---
ARCHIVO_LOTE = os.path.join("output_escaner_encodings", "lote_encodings.pkl")
ARCHIVO_PERFIL = os.path.join("output_comparador_caras", "perfil_objetivo.pkl")
CARPETA_SALIDA = "output_buscador_objetivo"
ARCHIVO_EXCEL = os.path.join(CARPETA_SALIDA, "resultado_busqueda.xlsx")

# --- MODO NIÑOS ---
# Activar cuando se buscan rostros de ninos (< ~12 años).
# Aplica restricciones adicionales porque los encodings de ninos
# son inherentemente mas similares entre si.
MODO_NINOS = True

# Tolerancia de coincidencia (ajustable sin re-escanear)
# NOTA: Para caras de NINOS, usar valores MAS BAJOS porque sus rasgos
# son menos diferenciados y producen mas falsos positivos.
# 0.30 = ultra estricto (minimas coincidencias, maxima precision)
# 0.33 = muy estricto (recomendado para ninos pequeños)
# 0.36 = estricto (recomendado para ninos en general)
# 0.40 = equilibrado (recomendado para adultos)
# 0.45 = flexible (mas coincidencias, puede incluir falsos positivos)
TOLERANCIA_BUSQUEDA = 0.35 if MODO_NINOS else 0.40

# Porcentaje minimo de muestras del perfil que deben reconocer la cara
# 0.5 = mayoria simple, 0.7 = mayoria fuerte, 1.0 = unanimidad
UMBRAL_VOTACION = 0.75 if MODO_NINOS else 0.6

# Desviacion estandar maxima de las distancias a las muestras del perfil.
# Si la std es alta, significa que unas muestras la reconocen y otras no
# = coincidencia ambigua = probablemente falso positivo.
# Valores bajos = mas estricto. Tipico: 0.04-0.06
MAX_STD_DISTANCIAS = 0.04 if MODO_NINOS else 0.06

# Score compuesto minimo (0-100). Combina todas las metricas en un solo
# numero. Requiere que el score sea >= este umbral para ser coincidencia.
MIN_SCORE_COMPUESTO = 55 if MODO_NINOS else 55


# =============================================================================
# FUNCIONES DE SCORING
# =============================================================================

def calcular_score_compuesto(dist_media, dist_mediana, porcentaje_votos, std_distancias,
                              tolerancia, num_muestras):
    """
    Calcula un score compuesto de 0 a 100 combinando multiples metricas.
    Score alto = mayor confianza de que es la persona correcta.
    
    Componentes:
      - score_distancia (40%): Que tan lejos esta la dist_media del umbral
      - score_votacion (30%): Porcentaje de muestras que votaron positivo
      - score_consistencia (20%): Inverso de la desviacion estandar
      - score_mediana (10%): Que tan lejos esta la mediana del umbral
    """
    # Score de distancia: 100 si dist=0, 0 si dist >= tolerancia
    score_distancia = max(0, (1 - dist_media / tolerancia)) * 100
    
    # Score de votacion: directo del porcentaje
    score_votacion = porcentaje_votos * 100
    
    # Score de consistencia: std baja = buena. std=0 -> 100, std>=0.08 -> 0
    score_consistencia = max(0, (1 - std_distancias / 0.08)) * 100
    
    # Score de mediana: similar a distancia pero con mediana
    score_mediana = max(0, (1 - dist_mediana / tolerancia)) * 100
    
    # Ponderacion: peso extra en distancia y votacion
    score_final = (
        score_distancia * 0.40 +
        score_votacion * 0.30 +
        score_consistencia * 0.20 +
        score_mediana * 0.10
    )
    
    return round(score_final, 1)


def clasificar_confianza(score):
    """Clasifica el score compuesto en niveles de confianza legibles."""
    if score >= 85:
        return "MUY ALTA"
    elif score >= 70:
        return "ALTA"
    elif score >= 55:
        return "MEDIA"
    elif score >= 40:
        return "BAJA"
    else:
        return "MUY BAJA"


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PASO 3/3: BUSCAR PERSONA EN LOTE DE IMAGENES")
    if MODO_NINOS:
        print("  [MODO NINOS ACTIVADO - Filtrado anti-falsos positivos reforzado]")
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

    # Calcular el medoid del perfil: la muestra mas cercana al centroide.
    # Es mas robusto que el promedio porque no se distorsiona con outliers.
    if len(todos_enc_persona) > 1:
        distancias_al_centroide = face_recognition.face_distance(
            todos_enc_persona, encoding_objetivo
        )
        idx_medoid = int(np.argmin(distancias_al_centroide))
        encoding_medoid = todos_enc_persona[idx_medoid]
        print(f"  Medoid seleccionado: muestra #{idx_medoid + 1} (dist al centroide: {distancias_al_centroide[idx_medoid]:.4f})")
    else:
        encoding_medoid = encoding_objetivo

    print(f"\n  Configuracion de busqueda:")
    print(f"    Tolerancia:           {TOLERANCIA_BUSQUEDA}")
    print(f"    Umbral votacion:      {UMBRAL_VOTACION:.0%}")
    print(f"    Max STD distancias:   {MAX_STD_DISTANCIAS}")
    print(f"    Min score compuesto:  {MIN_SCORE_COMPUESTO}")
    print(f"    Modo ninos:           {'SI' if MODO_NINOS else 'NO'}")
    print(f"  Origen del lote: {metadata_lote.get('carpeta_origen', 'desconocido')}")
    print("-" * 65)

    # --- Comparacion ---
    print(f"\n  Comparando {len(caras_lote)} caras contra '{nombre_persona}'...\n")
    tiempo_inicio = time.time()

    # Convertir todos los encodings del lote a una matriz numpy
    encodings_lote = np.array([cara["encoding"] for cara in caras_lote])

    # Pre-filtro rapido: distancia al promedio Y al medoid
    distancias_al_promedio = face_recognition.face_distance(encodings_lote, encoding_objetivo)
    distancias_al_medoid = face_recognition.face_distance(encodings_lote, encoding_medoid)

    # Margen para pre-filtro (un poco mas amplio para no perder verdaderos positivos)
    MARGEN_PREFILTRO = 0.08

    coincidencias = []
    revision_manual = []  # Coincidencias borderline para revision humana
    fotos_con_coincidencia = set()
    stats_filtros = {"prefiltro": 0, "distancia": 0, "votacion": 0, 
                     "consistencia": 0, "mediana": 0, "score": 0, "aceptadas": 0}

    for i, cara in enumerate(caras_lote):
        dist_al_promedio = distancias_al_promedio[i]
        dist_al_medoid = distancias_al_medoid[i]

        # --- CAPA 0: Pre-filtro rapido ---
        # Ambas distancias (promedio y medoid) deben estar dentro del margen
        if dist_al_promedio > TOLERANCIA_BUSQUEDA + MARGEN_PREFILTRO and \
           dist_al_medoid > TOLERANCIA_BUSQUEDA + MARGEN_PREFILTRO:
            stats_filtros["prefiltro"] += 1
            continue

        # --- Calcular metricas detalladas ---
        if len(todos_enc_persona) > 1:
            distancias_a_muestras = face_recognition.face_distance(
                todos_enc_persona, np.array(cara["encoding"])
            )
            dist_media = float(np.mean(distancias_a_muestras))
            dist_mediana = float(np.median(distancias_a_muestras))
            dist_minima = float(np.min(distancias_a_muestras))
            dist_maxima = float(np.max(distancias_a_muestras))
            std_distancias = float(np.std(distancias_a_muestras))

            # Votacion: cuantas muestras reconocen esta cara individualmente
            votos_positivos = int(np.sum(distancias_a_muestras <= TOLERANCIA_BUSQUEDA))
            porcentaje_votos = votos_positivos / len(todos_enc_persona)
        else:
            dist_media = float(dist_al_promedio)
            dist_mediana = float(dist_al_promedio)
            dist_minima = float(dist_al_promedio)
            dist_maxima = float(dist_al_promedio)
            std_distancias = 0.0
            votos_positivos = 1 if dist_al_promedio <= TOLERANCIA_BUSQUEDA else 0
            porcentaje_votos = 1.0 if dist_al_promedio <= TOLERANCIA_BUSQUEDA else 0.0

        # Calcular score compuesto
        score = calcular_score_compuesto(
            dist_media, dist_mediana, porcentaje_votos, std_distancias,
            TOLERANCIA_BUSQUEDA, num_muestras
        )

        confianza = clasificar_confianza(score)

        # Datos de esta cara para reportes
        datos_cara = {
            "Archivo": cara["nombre_archivo"],
            "Ruta Completa": cara["ruta_archivo"],
            "Distancia Media": round(dist_media, 4),
            "Distancia Mediana": round(dist_mediana, 4),
            "Distancia Minima": round(dist_minima, 4),
            "Distancia Maxima": round(dist_maxima, 4),
            "STD Distancias": round(std_distancias, 4),
            "Votos": f"{votos_positivos}/{len(todos_enc_persona)}",
            "Votos %": round(porcentaje_votos * 100, 1),
            "Score": score,
            "Confianza": confianza
        }

        # --- SISTEMA DE FILTRADO MULTICAPA ---
        filtro_fallido = None

        # CAPA 1: Distancia media
        if dist_media > TOLERANCIA_BUSQUEDA:
            filtro_fallido = "distancia"
            stats_filtros["distancia"] += 1

        # CAPA 2: Votacion por consenso
        if filtro_fallido is None and porcentaje_votos < UMBRAL_VOTACION:
            filtro_fallido = "votacion"
            stats_filtros["votacion"] += 1

        # CAPA 3: Consistencia (solo si hay 3+ muestras)
        if filtro_fallido is None and len(todos_enc_persona) >= 3:
            if std_distancias > MAX_STD_DISTANCIAS:
                filtro_fallido = "consistencia"
                stats_filtros["consistencia"] += 1

        # CAPA 4: Mediana (la mediana tambien debe pasar)
        if filtro_fallido is None and dist_mediana > TOLERANCIA_BUSQUEDA:
            filtro_fallido = "mediana"
            stats_filtros["mediana"] += 1

        # CAPA 5: Score compuesto minimo
        if filtro_fallido is None and score < MIN_SCORE_COMPUESTO:
            filtro_fallido = "score"
            stats_filtros["score"] += 1

        # --- Clasificar resultado ---
        if filtro_fallido is None:
            # TODAS las capas pasaron = coincidencia confirmada
            coincidencias.append(datos_cara)
            fotos_con_coincidencia.add(cara["ruta_archivo"])
            stats_filtros["aceptadas"] += 1
        elif dist_media <= TOLERANCIA_BUSQUEDA + 0.03 and porcentaje_votos >= 0.5:
            # Casi paso pero fallo en una capa = revision manual
            datos_cara["Filtro Fallido"] = filtro_fallido
            revision_manual.append(datos_cara)

    tiempo_busqueda = time.time() - tiempo_inicio

    # --- Mostrar resultados ---
    print(f"  Busqueda completada en {tiempo_busqueda:.2f}s")
    print(f"\n  RESULTADOS PARA: {nombre_persona}")
    print(f"  {'=' * 50}")
    print(f"  Coincidencias confirmadas: {len(coincidencias)}")
    print(f"  Revision manual (borderline): {len(revision_manual)}")
    print(f"  Fotos unicas con esta persona: {len(fotos_con_coincidencia)}")

    # Desglose de filtros
    print(f"\n  Desglose de filtrado:")
    print(f"    Descartadas por pre-filtro:     {stats_filtros['prefiltro']}")
    print(f"    Filtradas por distancia:         {stats_filtros['distancia']}")
    print(f"    Filtradas por votacion:          {stats_filtros['votacion']}")
    print(f"    Filtradas por consistencia (STD):{stats_filtros['consistencia']}")
    print(f"    Filtradas por mediana:           {stats_filtros['mediana']}")
    print(f"    Filtradas por score compuesto:   {stats_filtros['score']}")
    print(f"    ACEPTADAS:                       {stats_filtros['aceptadas']}")

    if len(coincidencias) > 0:
        # Ordenar por score (mayor = mejor coincidencia)
        coincidencias.sort(key=lambda x: x["Score"], reverse=True)

        # Estadisticas
        distancias = [c["Distancia Media"] for c in coincidencias]
        scores = [c["Score"] for c in coincidencias]
        print(f"\n  Estadisticas de coincidencias confirmadas:")
        print(f"    Distancia promedio: {np.mean(distancias):.4f}")
        print(f"    Mejor distancia:    {min(distancias):.4f}")
        print(f"    Peor distancia:     {max(distancias):.4f}")
        print(f"    Score promedio:     {np.mean(scores):.1f}")
        print(f"    Score minimo:       {min(scores):.1f}")

        # Mostrar las mejores coincidencias en consola
        print(f"\n  Top 10 mejores coincidencias:")
        for i, c in enumerate(coincidencias[:10], 1):
            print(f"    {i}. [{c['Confianza']}] {c['Archivo']} "
                  f"(score: {c['Score']}, dist: {c['Distancia Media']}, votos: {c['Votos']})")

    if len(revision_manual) > 0:
        revision_manual.sort(key=lambda x: x["Score"], reverse=True)
        print(f"\n  ⚠ Casos borderline para revision manual:")
        for i, c in enumerate(revision_manual[:5], 1):
            print(f"    {i}. {c['Archivo']} "
                  f"(score: {c['Score']}, dist: {c['Distancia Media']}, "
                  f"votos: {c['Votos']}, fallo: {c['Filtro Fallido']})")

    # --- Exportar a Excel ---
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    if len(coincidencias) > 0 or len(revision_manual) > 0:
        # Hoja 1: Coincidencias confirmadas
        df_coincidencias = pd.DataFrame(coincidencias) if coincidencias else pd.DataFrame()

        # Hoja 2: Revision manual (borderline)
        df_revision = pd.DataFrame(revision_manual) if revision_manual else pd.DataFrame()

        # Hoja 3: Resumen por foto unica (sin duplicados)
        fotos_unicas = {}
        for c in coincidencias:
            ruta = c["Ruta Completa"]
            if ruta not in fotos_unicas or c["Score"] > fotos_unicas[ruta]["Score"]:
                fotos_unicas[ruta] = {
                    "Archivo": c["Archivo"],
                    "Ruta Completa": ruta,
                    "Mejor Distancia": c["Distancia Media"],
                    "Score": c["Score"],
                    "Confianza": c["Confianza"]
                }

        fotos_resumen = sorted(fotos_unicas.values(), key=lambda x: x["Score"], reverse=True)
        df_resumen = pd.DataFrame(fotos_resumen)

        # Hoja 4: Todas las fotos del lote con indicador
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

        # Hoja 5: Configuracion usada (para reproducibilidad)
        config_data = {
            "Parametro": [
                "Tolerancia", "Umbral Votacion", "Max STD", "Min Score",
                "Modo Ninos", "Margen Prefiltro", "Muestras Perfil"
            ],
            "Valor": [
                TOLERANCIA_BUSQUEDA, UMBRAL_VOTACION, MAX_STD_DISTANCIAS,
                MIN_SCORE_COMPUESTO, MODO_NINOS, MARGEN_PREFILTRO, num_muestras
            ]
        }
        df_config = pd.DataFrame(config_data)

        # Guardar Excel
        with pd.ExcelWriter(ARCHIVO_EXCEL, engine='openpyxl') as writer:
            if not df_coincidencias.empty:
                df_coincidencias.to_excel(writer, sheet_name="Coincidencias", index=False)
            if not df_revision.empty:
                df_revision.to_excel(writer, sheet_name="Revision Manual", index=False)
            if not df_resumen.empty:
                df_resumen.to_excel(writer, sheet_name="Fotos Unicas", index=False)
            df_todas.to_excel(writer, sheet_name="Lote Completo", index=False)
            df_config.to_excel(writer, sheet_name="Configuracion", index=False)

        print(f"\n  Excel generado: {ARCHIVO_EXCEL}")
        if not df_coincidencias.empty:
            print(f"     -> Hoja 'Coincidencias' ({len(coincidencias)} filas)")
        if not df_revision.empty:
            print(f"     -> Hoja 'Revision Manual' ({len(revision_manual)} filas) ⚠")
        if not df_resumen.empty:
            print(f"     -> Hoja 'Fotos Unicas' ({len(fotos_resumen)} fotos)")
        print(f"     -> Hoja 'Lote Completo' ({len(todas_fotos)} fotos totales)")
        print(f"     -> Hoja 'Configuracion' (parametros usados)")

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

        print(f"\n  Imagenes confirmadas copiadas: {copiadas} en '{carpeta_objetivo}'")
        if errores_copia > 0:
            print(f"  {errores_copia} imagenes no se pudieron copiar (archivo no encontrado)")

        # --- Copiar imagenes borderline a subcarpeta de revision ---
        if len(revision_manual) > 0:
            carpeta_revision = os.path.join(carpeta_objetivo, "revision_manual")
            os.makedirs(carpeta_revision, exist_ok=True)

            copiadas_rev = 0
            errores_rev = 0
            rutas_revision = set(c["Ruta Completa"] for c in revision_manual)

            for ruta in rutas_revision:
                ruta_arreglada = ruta.replace('\\', os.sep).replace('/', os.sep)

                if not os.path.isfile(ruta_arreglada) and 'fotos_prueba' in ruta_arreglada:
                    partes = ruta_arreglada.split('fotos_prueba')
                    ruta_arreglada = os.path.join('fotos_prueba', partes[-1].lstrip(os.sep))

                if os.path.isfile(ruta_arreglada):
                    nombre_destino = os.path.basename(ruta_arreglada)
                    destino = os.path.join(carpeta_revision, nombre_destino)
                    if os.path.exists(destino):
                        base, ext = os.path.splitext(nombre_destino)
                        contador = 1
                        while os.path.exists(destino):
                            destino = os.path.join(carpeta_revision, f"{base}_{contador}{ext}")
                            contador += 1
                    try:
                        shutil.copy2(ruta_arreglada, destino)
                        copiadas_rev += 1
                    except Exception:
                        errores_rev += 1
                else:
                    errores_rev += 1

            print(f"  Imagenes borderline copiadas: {copiadas_rev} en '{carpeta_revision}'")
            if errores_rev > 0:
                print(f"  {errores_rev} imagenes borderline no se pudieron copiar")

    else:
        print(f"\n  No se encontraron coincidencias con tolerancia {TOLERANCIA_BUSQUEDA}.")
        print(f"  Sugerencias:")
        print(f"    - Aumentar TOLERANCIA_BUSQUEDA (ej: 0.40)")
        print(f"    - Desactivar MODO_NINOS si son adultos")
        print(f"    - Agregar mas fotos de referencia en definir_objetivo.py")
        print(f"    - Verificar que las fotos del perfil son claras y frontales")

    print(f"\n{'=' * 65}")
    print(f"  BUSQUEDA FINALIZADA")
    print(f"  Persona: {nombre_persona}")
    print(f"  Lote: {metadata_lote['total_imagenes']} imagenes ({len(caras_lote)} caras)")
    print(f"  Resultado: {len(coincidencias)} confirmadas + {len(revision_manual)} para revision")
    print(f"  Fotos unicas: {len(fotos_con_coincidencia)}")
    print(f"{'=' * 65}")
