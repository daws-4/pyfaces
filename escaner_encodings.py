import os
import sys
import seguridad
import time
import gc
import pickle
import traceback
from datetime import datetime
import face_recognition
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor

# =============================================================================
# PASO 1 DE 3: EXTRAER ENCODINGS DE UN LOTE DE IMAGENES
# =============================================================================
# Este script escanea todas las fotos de una carpeta, extrae los encodings
# faciales (vectores de 128 dimensiones) y los guarda en un archivo .pkl
# junto con la ruta exacta de cada foto.
#
# Este es el paso MAS LENTO (puede tardar 1+ hora). Solo necesitas
# ejecutarlo UNA VEZ por cada lote de fotos. Los otros scripts reusan
# el archivo .pkl generado aqui.
# =============================================================================

# --- CONFIGURACION ---
CARPETA_A_ESCANEAR = "fotos"
CARPETA_SALIDA = "output_escaner_encodings"
ARCHIVO_SALIDA = os.path.join(CARPETA_SALIDA, "lote_encodings.pkl")

# --- Parametros de deteccion ---
MODELO_DETECCION = "hog"    # "hog" = rapido en CPU, "cnn" = mas preciso pero lento
UPSAMPLE_VECES = 2          # 2 = detecta caras medianas y grandes
NUM_JITTERS = 20             # 20 = encodings muy estables
MODELO_ENCODING = "large"   # "large" = 68 puntos faciales

# --- Parametros de procesamiento ---
MAX_ANCHO = 1800             # Limitar resolucion para controlar memoria
TAMANO_LOTE = 20             # Imagenes por lote antes de liberar memoria
APLICAR_CLAHE = True

# --- Calculo automatico de workers segun RAM disponible ---
def calcular_workers():
    """Estima los workers seguros según la memoria libre en el contenedor."""
    import os
    memoria_por_worker_gb = 0.8  # RAM estimada por worker (HOG + MAX_ANCHO=1800)
    try:
        # En entornos Linux/Docker, inspecciona memoria verdaderamente asignada y disponible
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo') as f:
                for linea in f:
                    if linea.startswith('MemAvailable:'):
                        mem_kb = int(linea.split()[1])
                        mem_gb = mem_kb / (1024 * 1024)
                        # Reservar 0.5 GB para sistema
                        mem_disponible_gb = max(0.5, mem_gb - 0.5)
                        workers_por_ram = int(mem_disponible_gb / memoria_por_worker_gb)
                        cpus = os.cpu_count() or 4
                        resultado = max(1, min(workers_por_ram, cpus))
                        print(f"  [Auto-config] RAM disponible: {mem_gb:.1f} GB | CPUs: {cpus} | Workers: {resultado}")
                        return resultado
    except Exception:
        pass
    
    # Fallback seguro a la mitad de núcleos asignados si no reconoce meminfo
    cpus = os.cpu_count() or 2
    return max(1, int(cpus / 2))

NUM_WORKERS = calcular_workers()
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
INTENTAR_ROTACIONES = True   # Probar rotaciones EXIF si no detecta caras


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def barra_progreso(actual, total, ancho=40, tiempo_inicio=None, prefijo=""):
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


def aplicar_clahe_img(imagen_np):
    lab = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_ecualizado = clahe.apply(l_channel)
    lab_ecualizado = cv2.merge([l_ecualizado, a_channel, b_channel])
    return cv2.cvtColor(lab_ecualizado, cv2.COLOR_LAB2RGB)


def cargar_imagen_con_rotacion(ruta):
    """Carga imagen aplicando rotacion EXIF si es necesario."""
    try:
        img_pil = Image.open(ruta)
    except Exception:
        return None  # Archivo no se puede abrir en absoluto

    # Intentar rotacion EXIF (si falla, continuar sin rotar)
    try:
        from PIL import ExifTags
        exif = img_pil._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    if value == 3:
                        img_pil = img_pil.rotate(180, expand=True)
                    elif value == 6:
                        img_pil = img_pil.rotate(270, expand=True)
                    elif value == 8:
                        img_pil = img_pil.rotate(90, expand=True)
                    break
    except Exception:
        pass  # EXIF corrupto o ausente, continuar sin rotar

    # Convertir a RGB (si falla, continuar como esta)
    try:
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
    except Exception:
        try:
            img_pil = img_pil.convert('RGB')
        except Exception:
            return None  # No se puede convertir a RGB

    # Limitar resolucion
    try:
        ancho, alto = img_pil.size
        if MAX_ANCHO and ancho > MAX_ANCHO:
            ratio = MAX_ANCHO / ancho
            nuevo_alto = int(alto * ratio)
            img_pil = img_pil.resize((MAX_ANCHO, nuevo_alto), Image.LANCZOS)
    except Exception:
        pass  # Si falla el resize, usar la imagen tal como esta

    try:
        imagen_np = np.array(img_pil)
        del img_pil
        return imagen_np
    except Exception:
        return None


def procesar_foto(ruta_completa):
    """Procesa una foto y devuelve sus encodings."""
    nombre_archivo = os.path.basename(ruta_completa)
    try:
        imagen_np = cargar_imagen_con_rotacion(ruta_completa)
        if imagen_np is None:
            return (ruta_completa, nombre_archivo, None, 0, "No se pudo cargar la imagen")

        # CLAHE para mejorar deteccion
        if APLICAR_CLAHE:
            imagen_clahe = aplicar_clahe_img(imagen_np)
        else:
            imagen_clahe = imagen_np

        # Detectar caras
        ubicaciones = face_recognition.face_locations(
            imagen_clahe,
            number_of_times_to_upsample=UPSAMPLE_VECES,
            model=MODELO_DETECCION
        )

        # La imagen usada para encoding debe ser la MISMA usada para deteccion
        imagen_para_encoding = imagen_clahe

        # Si no detecto caras e INTENTAR_ROTACIONES, probar rotaciones manuales
        if len(ubicaciones) == 0 and INTENTAR_ROTACIONES:
            for angulo in [90, 180, 270]:
                img_rot = np.rot90(imagen_np, k=angulo // 90)
                if APLICAR_CLAHE:
                    img_rot_clahe = aplicar_clahe_img(img_rot)
                else:
                    img_rot_clahe = img_rot
                ubicaciones = face_recognition.face_locations(
                    img_rot_clahe,
                    number_of_times_to_upsample=UPSAMPLE_VECES,
                    model=MODELO_DETECCION
                )
                if len(ubicaciones) > 0:
                    imagen_para_encoding = img_rot_clahe if APLICAR_CLAHE else img_rot
                    break

        # Extraer encodings (usar la misma imagen donde se detectaron las caras)
        if len(ubicaciones) > 0:
            try:
                encodings = face_recognition.face_encodings(
                    imagen_para_encoding,
                    known_face_locations=ubicaciones,
                    num_jitters=NUM_JITTERS,
                    model=MODELO_ENCODING
                )
            except Exception:
                error_msg = traceback.format_exc()
                del imagen_np, imagen_clahe
                return (ruta_completa, nombre_archivo, None, 0, error_msg)
        else:
            encodings = []

        del imagen_np, imagen_clahe
        return (ruta_completa, nombre_archivo, encodings, len(ubicaciones), None)

    except Exception:
        error_msg = traceback.format_exc()
        return (ruta_completa, nombre_archivo, None, 0, error_msg)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    tiempo_inicio = time.time()

    print("=" * 65)
    print("  PASO 1/3: EXTRACCION MASIVA DE ENCODINGS FACIALES")
    print("=" * 65)
    print(f"  Carpeta: {CARPETA_A_ESCANEAR}")
    print(f"  Modelo: {MODELO_DETECCION.upper()} | Jitters: {NUM_JITTERS} | Upsample: {UPSAMPLE_VECES}x")
    print(f"  CLAHE: {'Si' if APLICAR_CLAHE else 'No'} | Max ancho: {MAX_ANCHO}px")
    print(f"  Workers: {NUM_WORKERS} | Lote: {TAMANO_LOTE}")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Recolectar fotos
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    fotos = []
    for ruta_actual, _, archivos in os.walk(CARPETA_A_ESCANEAR):
        for nombre in archivos:
            if os.path.splitext(nombre)[1].lower() in extensiones:
                fotos.append(os.path.join(ruta_actual, nombre))

    total_fotos = len(fotos)
    if total_fotos == 0:
        print(f"\n  No se encontraron imagenes en '{CARPETA_A_ESCANEAR}'.")
        sys.exit(1)

    print(f"\n  {total_fotos} imagenes encontradas. Iniciando extraccion...\n")

    # Procesar en lotes paralelos
    todas_las_caras = []  # Lista de {encoding, ruta, archivo}
    total_caras = 0
    errores = 0
    procesadas = 0

    fotos_fallidas = []  # Para reintentar secuencialmente si el pool crashea

    for i_lote in range(0, total_fotos, TAMANO_LOTE):
        lote = fotos[i_lote:i_lote + TAMANO_LOTE]
        pool_roto = False

        try:
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futuros = {executor.submit(procesar_foto, ruta): ruta for ruta in lote}

                for futuro in as_completed(futuros):
                    procesadas += 1
                    try:
                        ruta, nombre, encodings, num_caras, error_msg = futuro.result()
                        if encodings is None:
                            errores += 1
                            if error_msg:
                                print(f"  [{procesadas}/{total_fotos}] {nombre} -> ERROR: {error_msg.splitlines()[-1]}")
                            else:
                                print(f"  [{procesadas}/{total_fotos}] {nombre} -> ERROR")
                        elif len(encodings) > 0:
                            for enc in encodings:
                                todas_las_caras.append({
                                    "encoding": enc,
                                    "ruta_archivo": ruta,
                                    "nombre_archivo": nombre
                                })
                            total_caras += num_caras
                            print(f"  [{procesadas}/{total_fotos}] {nombre} -> OK ({num_caras} cara{'s' if num_caras > 1 else ''})")
                        else:
                            print(f"  [{procesadas}/{total_fotos}] {nombre} -> SIN CARA")
                    except BrokenExecutor:
                        pool_roto = True
                        # Recolectar fotos pendientes del lote para reintento
                        for f2, ruta2 in futuros.items():
                            if not f2.done() or f2 == futuro:
                                fotos_fallidas.append(ruta2)
                        print(f"\n  ⚠ Pool de procesos crasheo (memoria insuficiente). Reintentando pendientes secuencialmente...")
                        break
                    except Exception as e:
                        errores += 1
                        ruta_fallida = futuros[futuro]
                        print(f"  [{procesadas}/{total_fotos}] {os.path.basename(ruta_fallida)} -> ERROR: {e}")
        except BrokenExecutor:
            pool_roto = True
            fotos_fallidas.extend(lote)
            print(f"\n  ⚠ Pool de procesos crasheo al iniciar lote. Reintentando secuencialmente...")

        gc.collect()

    # Reintentar fotos fallidas de forma secuencial (sin paralelo)
    if fotos_fallidas:
        print(f"\n  Reintentando {len(fotos_fallidas)} imagenes de forma secuencial...")
        for ruta in fotos_fallidas:
            procesadas += 1
            try:
                ruta_r, nombre_r, encodings_r, num_caras_r, error_msg = procesar_foto(ruta)
                if encodings_r is None:
                    errores += 1
                    if error_msg:
                        print(f"  [{procesadas}/{total_fotos}] {nombre_r} -> ERROR: {error_msg.splitlines()[-1]}")
                    else:
                        print(f"  [{procesadas}/{total_fotos}] {nombre_r} -> ERROR")
                elif len(encodings_r) > 0:
                    for enc in encodings_r:
                        todas_las_caras.append({
                            "encoding": enc,
                            "ruta_archivo": ruta_r,
                            "nombre_archivo": nombre_r
                        })
                    total_caras += num_caras_r
                    print(f"  [{procesadas}/{total_fotos}] {nombre_r} -> OK ({num_caras_r} cara{'s' if num_caras_r > 1 else ''})")
                else:
                    print(f"  [{procesadas}/{total_fotos}] {nombre_r} -> SIN CARA")
            except Exception as e:
                errores += 1
                print(f"  [{procesadas}/{total_fotos}] {os.path.basename(ruta)} -> ERROR: {e}")
            gc.collect()

    print()

    # Guardar resultado
    datos_salida = {
        "caras": [],
        "metadata": {
            "total_imagenes": total_fotos,
            "total_caras": total_caras,
            "modelo": MODELO_DETECCION,
            "jitters": NUM_JITTERS,
            "upsample": UPSAMPLE_VECES,
            "max_ancho": MAX_ANCHO,
            "clahe": APLICAR_CLAHE,
            "fecha_creacion": time.strftime("%Y-%m-%d %H:%M:%S"),
            "carpeta_origen": os.path.abspath(CARPETA_A_ESCANEAR)
        }
    }

    for cara in todas_las_caras:
        datos_salida["caras"].append({
            "encoding": cara["encoding"].tolist(),
            "ruta_archivo": cara["ruta_archivo"],
            "nombre_archivo": cara["nombre_archivo"]
        })

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    with open(ARCHIVO_SALIDA, "wb") as f:
        pickle.dump(datos_salida, f)

    # Resumen
    tiempo_total = time.time() - tiempo_inicio
    minutos = int(tiempo_total // 60)
    segundos = int(tiempo_total % 60)

    print(f"\n{'=' * 65}")
    print(f"  EXTRACCION COMPLETADA EN {minutos}m {segundos}s")
    print(f"  {total_fotos} imagenes procesadas | {total_caras} caras extraidas")
    if errores > 0:
        print(f"  {errores} imagenes con error")
    print(f"\n  Archivo generado: {ARCHIVO_SALIDA}")
    print(f"  Tamano: {os.path.getsize(ARCHIVO_SALIDA) / 1024:.1f} KB")
    print(f"\n  Inicio: {datetime.fromtimestamp(tiempo_inicio).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Fin:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duracion: {minutos}m {segundos}s")
    print(f"\n  Siguiente paso: ejecutar comparador_caras.py")
    print(f"{'=' * 65}")