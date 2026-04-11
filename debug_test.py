"""Test with the EXACT procesar_foto from escaner_encodings.py"""
import sys
import os
import traceback

# Reproduce the exact same setup
sys.path.insert(0, '.')

# Import and run exactly like escaner_encodings.py does
CARPETA_A_ESCANEAR = "fotos"
MODELO_DETECCION = "hog"
UPSAMPLE_VECES = 2
NUM_JITTERS = 20
MODELO_ENCODING = "large"
MAX_ANCHO = 2400
NUM_WORKERS = 12
TAMANO_LOTE = 15
APLICAR_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
INTENTAR_ROTACIONES = True

import face_recognition
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# This is a KEY difference: import the function from the module vs defining it here
# The original script uses module-level globals. When ProcessPoolExecutor forks,
# the child processes need access to these globals.
# Let's try importing procesar_foto directly from escaner_encodings

print("Test 1: Import procesar_foto from escaner_encodings...")
try:
    from escaner_encodings import procesar_foto
    print("  Import OK")
except Exception:
    traceback.print_exc()
    print("\n  Falling back to direct definition test...")

# Find first 3 images
extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
fotos = []
for ruta_actual, _, archivos in os.walk(CARPETA_A_ESCANEAR):
    for nombre in archivos:
        if os.path.splitext(nombre)[1].lower() in extensiones:
            fotos.append(os.path.join(ruta_actual, nombre))
            if len(fotos) >= 3:
                break
    if len(fotos) >= 3:
        break

print(f"\nTest 2: Process {len(fotos)} images with ProcessPoolExecutor (workers=2)...")
print(f"  Images: {fotos}")

with ProcessPoolExecutor(max_workers=2) as executor:
    futuros = {executor.submit(procesar_foto, ruta): ruta for ruta in fotos}
    for futuro in as_completed(futuros):
        ruta_orig = futuros[futuro]
        try:
            ruta, nombre, encodings, num_caras = futuro.result(timeout=120)
            if encodings is None:
                print(f"  {nombre} -> ERROR (encodings is None)")
            elif len(encodings) > 0:
                print(f"  {nombre} -> OK ({num_caras} faces)")
            else:
                print(f"  {nombre} -> NO FACES")
        except Exception as e:
            print(f"  {os.path.basename(ruta_orig)} -> EXCEPTION: {e}")
            traceback.print_exc()

print("\nTest 3: Process same images with workers=12 (like original)...")
with ProcessPoolExecutor(max_workers=12) as executor:
    futuros = {executor.submit(procesar_foto, ruta): ruta for ruta in fotos}
    for futuro in as_completed(futuros):
        ruta_orig = futuros[futuro]
        try:
            ruta, nombre, encodings, num_caras = futuro.result(timeout=120)
            if encodings is None:
                print(f"  {nombre} -> ERROR (encodings is None)")
            elif len(encodings) > 0:
                print(f"  {nombre} -> OK ({num_caras} faces)")
            else:
                print(f"  {nombre} -> NO FACES")
        except Exception as e:
            print(f"  {os.path.basename(ruta_orig)} -> EXCEPTION: {e}")
            traceback.print_exc()

print("\n=== DONE ===")
