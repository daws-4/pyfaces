import os
import gc
import sys
import seguridad
import time
import face_recognition
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================================================================
# CONFIGURACIÓN — MÁXIMA PRECISIÓN PARA i5-12th Gen / 16GB RAM
# =============================================================================
CARPETA_A_ESCANEAR = "fotos"
ARCHIVO_TXT = "resultados_caras.txt"
ARCHIVO_EXCEL = "reporte_caras.xlsx"

# --- Parámetros de detección (MÁXIMA PRECISIÓN) ---
MODELO_DETECCION = "cnn"    # "cnn" = red neuronal profunda
UPSAMPLE_VECES = 2          # 2 = detecta caras pequeñas/lejanas
NUM_JITTERS = 10             # 10 = encodings más estables
MODELO_ENCODING = "large"   # "large" = 68 puntos faciales

# --- Parámetros de agrupación ---
TOLERANCIA = 0.45            # 0.45 = muy estricto

# --- Parámetros de video ---
MUESTREO_VIDEO_SEG = 0.5    # Cada 0.5 segundos
MAX_ANCHO_VIDEO = 1920      # Los frames de video se limitan a 1080p para controlar RAM

# --- Parámetros de procesamiento AJUSTADOS PARA 16GB RAM ---
MAX_ANCHO_FOTO = 2500       # Fotos se limitan a 2500px (altísimo, pero seguro)
NUM_WORKERS = 4              # CNN consume ~500MB-1GB por worker. 4 = seguro en 16GB.
TAMANO_LOTE = 10             # Procesar en lotes pequeños para liberar memoria

APLICAR_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def barra_progreso(actual, total, ancho=40, tiempo_inicio=None, prefijo=""):
    """Imprime una barra de progreso en la consola."""
    porcentaje = actual / total if total > 0 else 1
    llenas = int(ancho * porcentaje)
    barra = "█" * llenas + "░" * (ancho - llenas)
    texto = f"\r  {prefijo}[{barra}] {actual}/{total} ({porcentaje:.0%})"

    if tiempo_inicio and actual > 0:
        transcurrido = time.time() - tiempo_inicio
        restante = (transcurrido / actual * total) - transcurrido
        minutos = int(restante // 60)
        segundos = int(restante % 60)
        texto += f" | ~{minutos}m {segundos}s"

    print(texto, end="", flush=True)


def aplicar_clahe(imagen_np):
    """Ecualización adaptativa de histograma para mejorar contraste."""
    lab = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_ecualizado = clahe.apply(l_channel)
    lab_ecualizado = cv2.merge([l_ecualizado, a_channel, b_channel])
    return cv2.cvtColor(lab_ecualizado, cv2.COLOR_LAB2RGB)


def redimensionar_np(imagen_np, max_ancho):
    """Redimensiona un numpy array si excede el ancho máximo."""
    alto, ancho = imagen_np.shape[:2]
    if ancho > max_ancho:
        ratio = max_ancho / ancho
        nuevo_alto = int(alto * ratio)
        return cv2.resize(imagen_np, (max_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    return imagen_np


# =============================================================================
# FUNCIONES DE PROCESAMIENTO POR WORKER
# =============================================================================

def recolectar_archivos(carpeta):
    """Recorre la carpeta y clasifica archivos en fotos y videos."""
    fotos = []
    videos = []
    extensiones_foto = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    extensiones_video = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm'}

    for ruta_actual, _, archivos in os.walk(carpeta):
        for nombre_archivo in archivos:
            ext = os.path.splitext(nombre_archivo)[1].lower()
            ruta_completa = os.path.join(ruta_actual, nombre_archivo)
            if ext in extensiones_foto:
                fotos.append(ruta_completa)
            elif ext in extensiones_video:
                videos.append(ruta_completa)

    return fotos, videos


def procesar_foto(ruta_completa):
    """
    Procesa una foto con máxima precisión, controlando memoria.
    CNN + CLAHE + jitters + upsample, con resolución limitada a 2500px.
    """
    nombre_archivo = os.path.basename(ruta_completa)
    try:
        img_pil = Image.open(ruta_completa)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        # Limitar resolución para controlar memoria
        ancho_original, alto_original = img_pil.size
        if MAX_ANCHO_FOTO and ancho_original > MAX_ANCHO_FOTO:
            ratio = MAX_ANCHO_FOTO / ancho_original
            nuevo_alto = int(alto_original * ratio)
            img_pil = img_pil.resize((MAX_ANCHO_FOTO, nuevo_alto), Image.LANCZOS)

        imagen_np = np.array(img_pil)
        del img_pil

        # CLAHE para mejorar detección en fotos difíciles
        if APLICAR_CLAHE:
            imagen_clahe = aplicar_clahe(imagen_np)
        else:
            imagen_clahe = imagen_np

        # Detectar caras con CNN
        ubicaciones = face_recognition.face_locations(
            imagen_clahe,
            number_of_times_to_upsample=UPSAMPLE_VECES,
            model=MODELO_DETECCION
        )

        # Extraer encodings con alta precisión (sobre imagen original, no CLAHE)
        if len(ubicaciones) > 0:
            encodings = face_recognition.face_encodings(
                imagen_np,
                known_face_locations=ubicaciones,
                num_jitters=NUM_JITTERS,
                model=MODELO_ENCODING
            )
        else:
            encodings = []

        del imagen_np, imagen_clahe
        return ("foto", ruta_completa, nombre_archivo, encodings, len(ubicaciones))

    except Exception:
        return ("foto", ruta_completa, nombre_archivo, None, 0)


def procesar_video(ruta_completa):
    """
    Procesa un video con máxima precisión, controlando memoria.
    CNN en cada frame muestreado, con resolución limitada a 1920px.
    """
    nombre_archivo = os.path.basename(ruta_completa)
    try:
        video = cv2.VideoCapture(ruta_completa)
        fps = round(video.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duracion_seg = total_frames / fps if fps > 0 else 0

        intervalo = max(1, int(fps * MUESTREO_VIDEO_SEG))

        todos_los_encodings = []
        total_caras = 0
        frames_analizados = 0
        frame_count = 0

        while True:
            exito, frame = video.read()
            if not exito:
                break

            if frame_count % intervalo == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Limitar resolución del frame
                frame_rgb = redimensionar_np(frame_rgb, MAX_ANCHO_VIDEO)

                # CLAHE
                if APLICAR_CLAHE:
                    frame_procesado = aplicar_clahe(frame_rgb)
                else:
                    frame_procesado = frame_rgb

                # Detección con CNN
                ubicaciones = face_recognition.face_locations(
                    frame_procesado,
                    number_of_times_to_upsample=UPSAMPLE_VECES,
                    model=MODELO_DETECCION
                )

                if ubicaciones:
                    encodings = face_recognition.face_encodings(
                        frame_rgb,  # Encodings sobre el frame original
                        known_face_locations=ubicaciones,
                        num_jitters=NUM_JITTERS,
                        model=MODELO_ENCODING
                    )
                    todos_los_encodings.extend(encodings)
                    total_caras += len(ubicaciones)

                del frame_rgb, frame_procesado
                frames_analizados += 1

            frame_count += 1

        video.release()
        return ("video", ruta_completa, nombre_archivo, todos_los_encodings,
                duracion_seg, frames_analizados, total_caras)

    except Exception:
        return ("video", ruta_completa, nombre_archivo, None, 0, 0, 0)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    tiempo_total_inicio = time.time()

    print("=" * 65)
    print("   🔬 ESCÁNER HÍBRIDO — MÁXIMA PRECISIÓN (i5-12th / 16GB)")
    print("=" * 65)
    print(f"  📐 Modelo detección : {MODELO_DETECCION.upper()} (red neuronal)")
    print(f"  🎯 Modelo encoding  : {MODELO_ENCODING} (68 puntos)")
    print(f"  🔍 Upsample         : {UPSAMPLE_VECES}x")
    print(f"  🔄 Jitters          : {NUM_JITTERS}x")
    print(f"  📏 Tolerancia       : {TOLERANCIA}")
    print(f"  🖼️  Resolución foto  : máx {MAX_ANCHO_FOTO}px")
    print(f"  🎬 Resolución video : máx {MAX_ANCHO_VIDEO}px")
    print(f"  🎞️  Muestreo video   : cada {MUESTREO_VIDEO_SEG}s")
    print(f"  🎨 CLAHE            : {'Sí' if APLICAR_CLAHE else 'No'}")
    print(f"  ⚙️  Workers          : {NUM_WORKERS}")
    print(f"  📦 Tamaño de lote   : {TAMANO_LOTE}")
    print("=" * 65)

    # --- RECOLECTAR ARCHIVOS ---
    print(f"\n🔍 Buscando archivos en '{CARPETA_A_ESCANEAR}'...")
    fotos, videos = recolectar_archivos(CARPETA_A_ESCANEAR)
    total_fotos = len(fotos)
    total_videos = len(videos)
    total_archivos = total_fotos + total_videos

    if total_archivos == 0:
        print("⚠️ No se encontraron imágenes ni videos. Verifica la carpeta.")
        exit()

    print(f"  📸 {total_fotos} fotos | 🎥 {total_videos} videos encontrados")
    print(f"  🚀 Procesando con {NUM_WORKERS} workers en lotes de {TAMANO_LOTE}\n")
    print("⚠️  CNN es lento pero EXTREMADAMENTE preciso. Ten paciencia.\n")

    # =========================================================================
    # FASE 1: EXTRAER ENCODINGS EN PARALELO POR LOTES
    # =========================================================================
    resultados_fotos = []
    resultados_videos = []
    procesados_total = 0
    errores = 0
    tiempo_fase1 = time.time()

    # --- Procesar FOTOS en lotes ---
    if total_fotos > 0:
        print(f"  📷 Procesando {total_fotos} fotos...")
        for i_lote in range(0, total_fotos, TAMANO_LOTE):
            lote = fotos[i_lote:i_lote + TAMANO_LOTE]

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futuros = {executor.submit(procesar_foto, ruta): ruta for ruta in lote}

                for futuro in as_completed(futuros):
                    procesados_total += 1
                    try:
                        resultado = futuro.result()
                        if resultado[3] is None:
                            errores += 1
                        resultados_fotos.append(resultado)
                    except Exception:
                        ruta_fallida = futuros[futuro]
                        nombre_fallido = os.path.basename(ruta_fallida)
                        resultados_fotos.append(("foto", ruta_fallida, nombre_fallido, None, 0))
                        errores += 1

                    barra_progreso(procesados_total, total_archivos, tiempo_inicio=tiempo_fase1, prefijo="⏳ ")

            gc.collect()

        print()

    # --- Procesar VIDEOS en lotes ---
    if total_videos > 0:
        print(f"\n  🎥 Procesando {total_videos} videos...")
        # Videos se procesan de a 2 máximo (consumen mucha más memoria que las fotos)
        lote_video = max(1, min(2, NUM_WORKERS))
        for i_lote in range(0, total_videos, lote_video):
            lote = videos[i_lote:i_lote + lote_video]

            with ProcessPoolExecutor(max_workers=lote_video) as executor:
                futuros = {executor.submit(procesar_video, ruta): ruta for ruta in lote}

                for futuro in as_completed(futuros):
                    procesados_total += 1
                    try:
                        resultado = futuro.result()
                        if resultado[3] is None:
                            errores += 1
                        resultados_videos.append(resultado)
                    except Exception:
                        ruta_fallida = futuros[futuro]
                        nombre_fallido = os.path.basename(ruta_fallida)
                        resultados_videos.append(("video", ruta_fallida, nombre_fallido, None, 0, 0, 0))
                        errores += 1

                    barra_progreso(procesados_total, total_archivos, tiempo_inicio=tiempo_fase1, prefijo="🎬 ")

            gc.collect()

        print()

    tiempo_fase1_total = time.time() - tiempo_fase1
    print(f"\n  ✅ Fase 1 completada en {tiempo_fase1_total:.1f}s")
    if errores > 0:
        print(f"  ⚠️ {errores} archivos no pudieron ser procesados.")

    # =========================================================================
    # FASE 2: IDENTIFICAR PERSONAS CON PROMEDIADO DE ENCODINGS
    # =========================================================================
    print(f"\n🧠 Fase 2: Identificando personas...")
    tiempo_fase2 = time.time()

    personas_encodings = {}
    personas_promedio = {}
    state = {'contador_personas': 1}
    datos_para_excel = []

    def identificar_encoding(cara_encoding):
        """Identifica con promediado acumulativo para mayor robustez."""

        if len(personas_promedio) > 0:
            nombres = list(personas_promedio.keys())
            encodings_promedio = list(personas_promedio.values())

            distancias = face_recognition.face_distance(encodings_promedio, cara_encoding)
            mejor_indice = np.argmin(distancias)

            if distancias[mejor_indice] <= TOLERANCIA:
                nombre = nombres[mejor_indice]
                personas_encodings[nombre].append(cara_encoding)
                personas_promedio[nombre] = np.mean(personas_encodings[nombre], axis=0)
                return nombre

        nombre = f"Persona_{state['contador_personas']}"
        personas_encodings[nombre] = [cara_encoding]
        personas_promedio[nombre] = cara_encoding
        state['contador_personas'] += 1
        return nombre

    # --- Procesar resultados de FOTOS ---
    resultados_fotos.sort(key=lambda x: x[1])
    for _, ruta_completa, nombre_archivo, encodings, num_caras in resultados_fotos:
        if encodings is None:
            datos_para_excel.append({
                "Archivo": nombre_archivo,
                "Tipo": "Foto",
                "Ruta Completa": ruta_completa,
                "Caras Encontradas": 0,
                "Personas Identificadas": "❌ Error al leer"
            })
            continue

        nombres = set()
        for enc in encodings:
            nombres.add(identificar_encoding(enc))

        texto = ", ".join(sorted(nombres)) if nombres else "Ninguna cara detectada"
        datos_para_excel.append({
            "Archivo": nombre_archivo,
            "Tipo": "Foto",
            "Ruta Completa": ruta_completa,
            "Caras Encontradas": num_caras,
            "Personas Identificadas": texto
        })

    # --- Procesar resultados de VIDEOS ---
    resultados_videos.sort(key=lambda x: x[1])
    for resultado in resultados_videos:
        _, ruta_completa, nombre_archivo, encodings, duracion, frames_analizados, total_caras = resultado

        if encodings is None:
            datos_para_excel.append({
                "Archivo": nombre_archivo,
                "Tipo": "Vídeo",
                "Ruta Completa": ruta_completa,
                "Caras Encontradas": 0,
                "Personas Identificadas": "❌ Error al leer"
            })
            continue

        nombres = set()
        for enc in encodings:
            nombres.add(identificar_encoding(enc))

        texto = ", ".join(sorted(nombres)) if nombres else "Ninguna cara detectada"
        minutos_vid = int(duracion // 60)
        segundos_vid = int(duracion % 60)
        datos_para_excel.append({
            "Archivo": nombre_archivo,
            "Tipo": f"Vídeo ({minutos_vid}m {segundos_vid}s, {frames_analizados} frames)",
            "Ruta Completa": ruta_completa,
            "Caras Encontradas": total_caras,
            "Personas Identificadas": texto
        })

    tiempo_fase2_total = time.time() - tiempo_fase2
    total_personas = state['contador_personas'] - 1
    print(f"  ✅ Fase 2 completada en {tiempo_fase2_total:.1f}s")
    print(f"  👤 Total de personas únicas: {total_personas}")

    # Mostrar confianza por persona
    if 0 < total_personas <= 50:
        print(f"\n  📊 Muestras por persona:")
        for persona in sorted(personas_encodings.keys(),
                              key=lambda x: int(x.split("_")[1]) if "_" in x else 0):
            n = len(personas_encodings[persona])
            confianza = "🟢" if n >= 5 else "🟡" if n >= 2 else "🔴"
            print(f"     {confianza} {persona}: {n} muestras")

    # =========================================================================
    # FASE 3: EXPORTAR RESULTADOS
    # =========================================================================
    print("\n💾 Guardando reportes...")

    if len(datos_para_excel) > 0:
        # TXT
        with open(ARCHIVO_TXT, "w", encoding="utf-8") as archivo_txt:
            archivo_txt.write("=" * 65 + "\n")
            archivo_txt.write("  REPORTE DE RECONOCIMIENTO FACIAL — MÁXIMA PRECISIÓN\n")
            archivo_txt.write("=" * 65 + "\n\n")
            archivo_txt.write(f"Modelo: {MODELO_DETECCION.upper()} | Encoding: {MODELO_ENCODING} | ")
            archivo_txt.write(f"Jitters: {NUM_JITTERS} | Tolerancia: {TOLERANCIA}\n")
            archivo_txt.write(f"Upsample: {UPSAMPLE_VECES}x | CLAHE: {'Sí' if APLICAR_CLAHE else 'No'}\n")
            archivo_txt.write(f"Personas únicas: {total_personas}\n\n")
            archivo_txt.write("-" * 65 + "\n\n")
            for dato in datos_para_excel:
                archivo_txt.write(
                    f"[{dato['Tipo']}] {dato['Ruta Completa']}\n"
                    f"  Caras: {dato['Caras Encontradas']} | Personas: {dato['Personas Identificadas']}\n\n"
                )
            archivo_txt.write("-" * 65 + "\n")
            archivo_txt.write(f"\nRESUMEN: {total_fotos} fotos + {total_videos} videos | "
                              f"{total_personas} personas únicas\n")
            archivo_txt.write(f"Tiempo total: {time.time() - tiempo_total_inicio:.1f}s\n")
        print(f"  ✔️ TXT creado: {ARCHIVO_TXT}")

        # Excel
        df = pd.DataFrame(datos_para_excel)
        df.to_excel(ARCHIVO_EXCEL, index=False)
        print(f"  ✔️ Excel creado: {ARCHIVO_EXCEL}")
    else:
        print("  ⚠️ No se encontraron archivos multimedia válidos.")

    # --- RESUMEN FINAL ---
    tiempo_total = time.time() - tiempo_total_inicio
    minutos = int(tiempo_total // 60)
    segundos = int(tiempo_total % 60)
    print(f"\n{'=' * 65}")
    print(f"  🎉 PROCESO FINALIZADO EN {minutos}m {segundos}s")
    print(f"  📸 {total_fotos} fotos | 🎥 {total_videos} videos | 👤 {total_personas} personas únicas")
    print(f"{'=' * 65}")