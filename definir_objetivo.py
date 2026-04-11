import os
import sys
import seguridad
import time
import pickle
import face_recognition
import numpy as np
import cv2
from PIL import Image

# =============================================================================
# PASO 2 DE 3: CREAR PERFIL DE UNA PERSONA ESPECIFICA
# =============================================================================
# Este script toma una carpeta con fotos donde aparece UNA SOLA PERSONA
# (puede ser 1 foto o muchas fotos de la misma persona).
# Extrae su encoding facial promedio y lo guarda como un "perfil objetivo"
# que el paso 3 usara para buscar coincidencias en el lote principal.
#
# REQUISITOS:
#   - Cada foto debe tener EXACTAMENTE 1 cara visible
#   - Todas las fotos deben ser de la MISMA persona
#   - Minimo 1 foto, recomendado 3-5 para mayor precision
# =============================================================================

# --- CONFIGURACION ---
CARPETA_PERSONA = "persona_objetivo"  # Carpeta con fotos de la persona a buscar
CARPETA_SALIDA = "output_comparador_caras"
ARCHIVO_SALIDA = os.path.join(CARPETA_SALIDA, "perfil_objetivo.pkl")
NOMBRE_PERSONA = ""  # Dejar vacio para que pregunte al ejecutar

# --- Parametros de deteccion (mismos que escaner_encodings.py) ---
MODELO_DETECCION = "hog"
UPSAMPLE_VECES = 2          # 2 = mismo valor que escaner_encodings.py para consistencia
NUM_JITTERS = 30            # 30 = mas alto para perfil objetivo (referencia base, pocas fotos)
MODELO_ENCODING = "large"
MAX_ANCHO = 2400
APLICAR_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)


# =============================================================================
# FUNCIONES
# =============================================================================

def aplicar_clahe_img(imagen_np):
    lab = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_ecualizado = clahe.apply(l_channel)
    lab_ecualizado = cv2.merge([l_ecualizado, a_channel, b_channel])
    return cv2.cvtColor(lab_ecualizado, cv2.COLOR_LAB2RGB)


def cargar_imagen(ruta):
    """Carga imagen con rotacion EXIF."""
    try:
        img_pil = Image.open(ruta)
        from PIL import ExifTags
        try:
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
        except (AttributeError, KeyError):
            pass

        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        ancho, alto = img_pil.size
        if MAX_ANCHO and ancho > MAX_ANCHO:
            ratio = MAX_ANCHO / ancho
            nuevo_alto = int(alto * ratio)
            img_pil = img_pil.resize((MAX_ANCHO, nuevo_alto), Image.LANCZOS)

        imagen_np = np.array(img_pil)
        del img_pil
        return imagen_np
    except Exception as e:
        print(f"    Error cargando {ruta}: {e}")
        return None


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PASO 2/3: CREAR PERFIL DE PERSONA OBJETIVO")
    print("=" * 65)

    # Verificar carpeta
    if not os.path.isdir(CARPETA_PERSONA):
        print(f"\n  La carpeta '{CARPETA_PERSONA}' no existe.")
        print(f"  Crea la carpeta y coloca fotos de la persona que quieres buscar.")
        print(f"  Cada foto debe mostrar EXACTAMENTE 1 cara.")
        sys.exit(1)

    # Pedir nombre si no esta definido
    nombre = NOMBRE_PERSONA
    if not nombre:
        nombre = input("\n  Nombre de la persona (ej: Juan, Maria): ").strip()
        if not nombre:
            nombre = "Persona_Objetivo"

    print(f"\n  Persona: {nombre}")
    print(f"  Carpeta: {CARPETA_PERSONA}")
    print(f"  Modelo: {MODELO_DETECCION.upper()} | Jitters: {NUM_JITTERS}")
    print("-" * 65)

    # Recolectar fotos
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    fotos = []
    for nombre_archivo in os.listdir(CARPETA_PERSONA):
        if os.path.splitext(nombre_archivo)[1].lower() in extensiones:
            fotos.append(os.path.join(CARPETA_PERSONA, nombre_archivo))

    if len(fotos) == 0:
        print(f"\n  No se encontraron imagenes en '{CARPETA_PERSONA}'.")
        sys.exit(1)

    print(f"\n  {len(fotos)} fotos encontradas. Procesando...\n")

    # Procesar cada foto
    encodings_validos = []
    fotos_usadas = []

    for i, ruta in enumerate(fotos, 1):
        nombre_foto = os.path.basename(ruta)
        print(f"  [{i}/{len(fotos)}] {nombre_foto}...", end=" ", flush=True)

        imagen_np = cargar_imagen(ruta)
        if imagen_np is None:
            print("ERROR al cargar")
            continue

        # CLAHE
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

        if len(ubicaciones) == 0:
            print("SIN CARAS detectadas (saltando)")
            continue

        if len(ubicaciones) > 1:
            print(f"MULTIPLE CARAS detectadas ({len(ubicaciones)}) - SALTANDO")
            print(f"           Esta foto tiene mas de 1 persona. Usa fotos con 1 sola cara.")
            continue

        # Extraer encoding (exactamente 1 cara)
        encodings = face_recognition.face_encodings(
            imagen_np,
            known_face_locations=ubicaciones,
            num_jitters=NUM_JITTERS,
            model=MODELO_ENCODING
        )

        if len(encodings) > 0:
            encodings_validos.append(encodings[0])
            fotos_usadas.append(nombre_foto)
            print("OK - cara extraida")
        else:
            print("ERROR al extraer encoding")

    print()

    if len(encodings_validos) == 0:
        print("  No se pudo extraer ningun encoding valido.")
        print("  Asegurate de que las fotos muestren claramente UNA cara.")
        sys.exit(1)

    # Calcular encoding promedio
    encoding_promedio = np.mean(encodings_validos, axis=0)

    # Verificar consistencia: si las fotos son de la misma persona,
    # las distancias entre sus encodings deben ser bajas
    if len(encodings_validos) >= 2:
        print(f"  Verificando consistencia de {len(encodings_validos)} muestras...")
        
        # Calcular distancia de cada muestra al centroide (promedio)
        distancias_al_centroide = []
        for enc in encodings_validos:
            d = np.linalg.norm(enc - encoding_promedio)
            distancias_al_centroide.append(d)
        
        # Calcular distancias internas (entre pares)
        distancias_internas = []
        for i in range(len(encodings_validos)):
            for j in range(i + 1, len(encodings_validos)):
                d = np.linalg.norm(encodings_validos[i] - encodings_validos[j])
                distancias_internas.append(d)

        dist_max = max(distancias_internas)
        dist_promedio = np.mean(distancias_internas)
        dist_std = np.std(distancias_internas)

        print(f"  Distancia promedio entre muestras: {dist_promedio:.4f}")
        print(f"  Distancia maxima entre muestras:   {dist_max:.4f}")
        print(f"  Desviacion estandar:               {dist_std:.4f}")

        # Deteccion de outliers: muestras muy lejanas al centroide
        media_dist_centroide = np.mean(distancias_al_centroide)
        std_dist_centroide = np.std(distancias_al_centroide)
        
        outliers = []
        if len(encodings_validos) >= 3 and std_dist_centroide > 0:
            for idx, d in enumerate(distancias_al_centroide):
                if d > media_dist_centroide + 2 * std_dist_centroide:
                    outliers.append((idx, fotos_usadas[idx], d))
        
        if outliers:
            print(f"\n  ⚠ Se detectaron {len(outliers)} muestra(s) atipica(s):")
            for idx, foto, d in outliers:
                print(f"    - {foto} (distancia al centroide: {d:.4f})")
            print(f"  Promedio de distancia al centroide: {media_dist_centroide:.4f}")
            print(f"  Estas fotos podrian ser de mala calidad o de otra persona.")
            respuesta = input("  Excluir outliers del perfil? (s/n): ").strip().lower()
            if respuesta == 's':
                indices_excluir = {idx for idx, _, _ in outliers}
                encodings_validos = [e for i, e in enumerate(encodings_validos) if i not in indices_excluir]
                fotos_excluidas = [f for idx, f, _ in outliers]
                fotos_usadas = [f for i, f in enumerate(fotos_usadas) if i not in indices_excluir]
                print(f"  Excluidas: {', '.join(fotos_excluidas)}")
                print(f"  Recalculando con {len(encodings_validos)} muestras...")
                encoding_promedio = np.mean(encodings_validos, axis=0)

        if dist_max > 0.6:
            print()
            print("  ADVERTENCIA: Algunas fotos podrian NO ser de la misma persona.")
            print("  La distancia maxima entre muestras es alta (> 0.6).")
            print("  Revisa que todas las fotos sean del mismo individuo.")
            respuesta = input("  Continuar de todas formas? (s/n): ").strip().lower()
            if respuesta != 's':
                print("  Abortado por el usuario.")
                sys.exit(0)
        else:
            print("  Consistencia: BUENA - todas las muestras son similares entre si.\n")

    # Calcular encoding mediana (mas robusto contra outliers)
    encoding_mediana = np.median(encodings_validos, axis=0) if len(encodings_validos) > 1 else encoding_promedio
    
    # Calcular desviacion estandar por dimension (indica que dimensiones varian mas)
    encoding_std = np.std(encodings_validos, axis=0) if len(encodings_validos) > 1 else np.zeros(128)

    # Guardar perfil
    perfil = {
        "nombre": nombre,
        "encoding_promedio": encoding_promedio,
        "encoding_mediana": encoding_mediana.tolist(),
        "encoding_std": encoding_std.tolist(),
        "todos_encodings": [e.tolist() for e in encodings_validos],
        "fotos_usadas": fotos_usadas,
        "num_muestras": len(encodings_validos),
        "metadata": {
            "modelo": MODELO_DETECCION,
            "jitters": NUM_JITTERS,
            "upsample": UPSAMPLE_VECES,
            "fecha_creacion": time.strftime("%Y-%m-%d %H:%M:%S"),
            "carpeta_origen": os.path.abspath(CARPETA_PERSONA)
        }
    }

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    with open(ARCHIVO_SALIDA, "wb") as f:
        pickle.dump(perfil, f)

    print(f"{'=' * 65}")
    print(f"  PERFIL CREADO EXITOSAMENTE")
    print(f"  Persona: {nombre}")
    print(f"  Muestras: {len(encodings_validos)} de {len(fotos)} fotos")
    print(f"  Fotos usadas: {', '.join(fotos_usadas)}")
    print(f"  Archivo: {ARCHIVO_SALIDA}")
    print(f"\n  Siguiente paso: ejecutar buscador_caras.py")
    print(f"{'=' * 65}")
