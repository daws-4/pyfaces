# py_faces — Reconocimiento Facial por Lotes

Sistema de reconocimiento facial diseñado para analizar grandes lotes de fotografías, extraer las huellas faciales (encodings) de cada persona detectada, y luego buscar a una persona específica dentro de todo el lote.

## Descripción

El proyecto funciona en **3 pasos independientes**, cada uno con su propio script. La idea principal es separar el proceso pesado (extraer encodings de fotos) del proceso liviano (comparar y buscar), de modo que el escaneo masivo solo se ejecuta **una vez** y las búsquedas posteriores son **instantáneas**.

### Arquitectura

```
┌─────────────────────────┐
│  PASO 1 (una sola vez)  │
│  escaner_encodings.py   │──────► output_escaner_encodings/
│  ~1 hora en CPU         │           └── lote_encodings.pkl
└─────────────────────────┘
                                              │
┌─────────────────────────┐                   │
│  PASO 2 (por persona)   │                   │
│  definir_objetivo.py    │──────► output_comparador_caras/
│  ~1 minuto              │           └── perfil_objetivo.pkl
└─────────────────────────┘                   │
                                              │
┌─────────────────────────┐                   │
│  PASO 3 (instantáneo)   │◄─────────────────┘
│  buscador_objetivo.py   │──────► output_buscador_objetivo/
│  ~2 segundos            │           ├── resultado_busqueda.xlsx
└─────────────────────────┘           └── {nombre_persona}/
                                              └── copias de fotos
```

## Instalación (Windows)

### Hardware recomendado

- CPU: Intel i5 12va gen o superior
- RAM: 16 GB mínimo
- GPU: No requerida (usa modelo HOG optimizado para CPU)

### Paso 1 — Python (la base)

No uses la versión más nueva (como la 3.14). Usa una versión madura y estable donde las librerías ya estén optimizadas.

1. Descarga e instala **Python 3.12** desde [python.org](https://www.python.org/downloads/).
2. **IMPORTANTE:** En la primera pantalla del instalador, marca la casilla **"Add Python to PATH"** (abajo del todo). Si no haces esto, ningún comando funcionará.

### Paso 2 — Compilador C++ (Visual Studio)

La librería `dlib` hace cálculos matemáticos intensivos y necesita C++ para compilarse correctamente.

1. Descarga [Visual Studio Community 2022](https://visualstudio.microsoft.com/es/downloads/).
2. En el instalador, marca la carga de trabajo: **"Desarrollo para el escritorio con C++"**.
3. En el panel derecho (Detalles), asegúrate de que estén marcados:
   - **MSVC v143** (Herramientas de compilación)
   - **SDK de Windows** (10 u 11)
   - **Herramientas de CMake de C++ para Windows** (vital)
4. Instala y **reinicia la computadora**.

### Paso 3 — dlib precompilado (el atajo)

Normalmente Python intentaría construir `dlib` desde cero (lo cual tarda mucho y a veces falla). El atajo es instalar una versión ya compilada:

```powershell
python -m pip install dlib-bin
```

### Paso 4 — Modelos de IA (el "cerebro")

Para evitar errores de modelos faltantes, forzamos la instalación directa:

```powershell
python -m pip install --force-reinstall face-recognition-models
```

### Paso 5 — Dependencias (herramientas de apoyo)

Instalamos de un solo golpe todo lo que el programa necesita para leer fotos, manejar arreglos matemáticos, procesar video y exportar a Excel:

```powershell
python -m pip install numpy pillow click opencv-python pandas openpyxl
```

### Paso 6 — face_recognition (la librería final)

Como ya le dimos a Python todo el trabajo pesado precompilado, instalamos `face_recognition` bloqueando su instinto de querer descargar dependencias por su cuenta:

```powershell
python -m pip install --no-deps face_recognition
```

## Paso a paso

### Paso 1 — Escanear el lote de fotos

Este es el paso más lento. Extrae los vectores faciales (encodings) de todas las fotos y los guarda en un archivo `.pkl`.

1. Coloca todas las fotos a analizar dentro de la carpeta `fotos_prueba/` (acepta subcarpetas).
2. Ejecuta:

```bash
python escaner_encodings.py
```

3. El script mostrará el progreso foto por foto:

```
[1/485] IMG_8542.JPG -> OK (2 caras)
[2/485] IMG_8543.JPG -> OK (1 cara)
[3/485] IMG_8544.JPG -> SIN CARA
```

4. Al finalizar, se genera `output_escaner_encodings/lote_encodings.pkl`.

**Este paso solo necesitas ejecutarlo una vez por cada lote de fotos.**

---

### Paso 2 — Definir la persona a buscar

Crea el perfil facial de la persona que quieres encontrar en el lote.

1. Crea (o limpia) la carpeta `persona_objetivo/`.
2. Coloca dentro fotos donde aparezca **únicamente esa persona** (1 sola cara por foto). Se recomiendan 3-5 fotos frontales y claras.
3. Ejecuta:

```bash
python definir_objetivo.py
```

4. El script pedirá el nombre de la persona y procesará cada foto:

```
Nombre de la persona (ej: Juan, Maria): graduando_1

[1/5] IMG_9073.JPG... OK - cara extraida
[2/5] IMG_9074.JPG... MULTIPLE CARAS detectadas (2) - SALTANDO
[3/5] IMG_9075.JPG... OK - cara extraida
```

5. Al finalizar, se genera `output_comparador_caras/perfil_objetivo.pkl`.

**Repite este paso cada vez que quieras buscar a una persona diferente.**

---

### Paso 3 — Buscar la persona en el lote

Compara el perfil del paso 2 contra todos los encodings del paso 1 y genera el reporte.

1. Ejecuta:

```bash
python buscador_objetivo.py
```

2. El script automáticamente carga los archivos de los pasos anteriores y muestra los resultados:

```
Coincidencias encontradas: 28
Fotos unicas con esta persona: 28

Top 10 mejores coincidencias:
  1. [MUY ALTA] IMG_9076.JPG (dist: 0.1454)
  2. [MUY ALTA] IMG_9084.JPG (dist: 0.1467)
```

3. Se genera:
   - `output_buscador_objetivo/resultado_busqueda.xlsx` — Excel con 3 hojas (coincidencias, fotos únicas, lote completo)
   - `output_buscador_objetivo/{nombre_persona}/` — Carpeta con copias de todas las fotos donde aparece la persona

**Este paso es instantáneo (~2 segundos). Puedes ejecutarlo múltiples veces cambiando la tolerancia.**

## Estructura de carpetas

```
py_faces/
├── escaner_encodings.py          # Paso 1: extraer encodings
├── definir_objetivo.py           # Paso 2: crear perfil de persona
├── buscador_objetivo.py          # Paso 3: buscar persona en lote
├── escaner_videos.py             # Script independiente para fotos + videos
├── fotos_prueba/                 # Carpeta con las fotos a analizar
├── persona_objetivo/             # Fotos de la persona a buscar (1 cara c/u)
├── output_escaner_encodings/     # Salida del paso 1
│   └── lote_encodings.pkl
├── output_comparador_caras/      # Salida del paso 2
│   └── perfil_objetivo.pkl
└── output_buscador_objetivo/     # Salida del paso 3
    ├── resultado_busqueda.xlsx
    └── {nombre_persona}/         # Copias de fotos con coincidencia
```

## Configuración

Cada script tiene parámetros ajustables en su sección `CONFIGURACION`:

### escaner_encodings.py

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `MODELO_DETECCION` | `"hog"` | `"hog"` = rápido en CPU, `"cnn"` = más preciso pero lento |
| `NUM_JITTERS` | `20` | Más jitters = encodings más estables, pero más lento |
| `UPSAMPLE_VECES` | `2` | `2` = detecta caras medianas/grandes. `1` = menos falsos positivos |
| `NUM_WORKERS` | `12` | Workers paralelos. Ajustar según núcleos de CPU |
| `MAX_ANCHO` | `2400` | Resolución máxima de procesamiento (px) |

### definir_objetivo.py

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `UPSAMPLE_VECES` | `1` | `1` para evitar detectar caras fantasma en fotos de primer plano |
| `NUM_JITTERS` | `20` | Mismo que el escáner para consistencia |

### buscador_objetivo.py

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `TOLERANCIA_BUSQUEDA` | `0.45` | `0.35` = muy estricto, `0.45` = equilibrado, `0.50` = flexible |

## Estados de las fotos

| Estado | Significado |
|--------|-------------|
| **OK (N caras)** | Imagen leída correctamente, se detectaron N rostros |
| **SIN CARA** | Imagen leída correctamente, pero no se detectó ningún rostro |
| **ERROR** | El archivo no se pudo abrir (corrupto, formato no soportado) |

## Niveles de confianza

| Nivel | Distancia | Significado |
|-------|-----------|-------------|
| **MUY ALTA** | ≤ 0.35 | Coincidencia casi segura |
| **ALTA** | 0.35 - 0.40 | Coincidencia confiable |
| **MEDIA** | 0.40 - 0.45 | Probable coincidencia, verificar manualmente |
| **BAJA** | > 0.45 | Coincidencia débil, posible falso positivo |

## Notas técnicas

- Los **encodings faciales** son vectores de 128 números de punto flotante que representan la "huella digital" de un rostro. Son generados por la librería `dlib` a través de `face_recognition`.
- El archivo `.pkl` (pickle) es un formato binario de Python que preserva la precisión completa de los vectores numéricos.
- El paso 1 es el cuello de botella porque cada imagen requiere detección facial (HOG/CNN) y cálculo de encoding con múltiples jitters. Los pasos 2 y 3 solo hacen operaciones matemáticas sobre vectores ya calculados.


.
