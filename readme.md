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
│  buscador_objetivo.py   │──────► personas_encontradas/
│  ~2 segundos            │           ├── resultado_busqueda.xlsx
└─────────────────────────┘           └── {nombre_persona}/
                                              └── copias de fotos
```

## Instalación (Recomendado: Docker)

Para evitar instalar herramientas de compilación pesadas (C++, Visual Studio, CMake), el proyecto ahora se ejecuta dentro de un contenedor Docker aislado. Esto garantiza que todos los programas funcionen independientemente de tu configuración del sistema.

### Pre-requisitos

1. **Docker Desktop:** Descarga e instala [Docker Desktop](https://www.docker.com/products/docker-desktop/).
   - Asegúrate de dejarlo ejecutándose en segundo plano (verás el ícono de la ballena cerca del reloj en la barra de tareas).
2. **Hardware recomendado:** CPU Intel i5 12va gen o superior, 16 GB de RAM.

### Instalación en 1 clic

1. Asegúrate de que Docker Desktop esté abierto.
2. Da **doble clic** en el archivo `0_construir_imagen.bat`.
3. Se abrirá una ventana negra que descargará el entorno de Linux, compilará C++ y preparará las dependencias automáticas.
   - *Nota: Este proceso toma alrededor de 10-15 minutos la primera vez. Una vez terminado, no necesitarás volver a hacerlo.*

### Asignación de Recursos para Docker (Importante para otros equipos)

Si vas a replicar el proyecto en otras computadoras, **Docker Desktop nativo en Windows (WSL2)** por defecto limita el uso de memoria a un 50% de la RAM total. Ya que el reconocimiento y las rutinas de mejora de imagen consumen mucha memoria, la extracción de encodings puede cerrarse abruptamente (crash the "Workers").

Para configurarlo de forma correcta en otros ordenadores y aprovechar toda su capacidad, sigue estos pasos:

1. **Memoria Base y Procesadores (Nivel Windows)**
   Debes crear un archivo llamado `.wslconfig` en la carpeta de tu usuario (generalmente en `C:\Users\TU_USUARIO\.wslconfig`) con el siguiente contenido:
   ```ini
   [wsl2]
   memory=10GB
   processors=10
   ```
   *Ajusta `10GB` y `10` según la capacidad de tu nuevo equipo. (No excedas ni el 80% de tu memoria ni el total de núcleos para no congelar tu PC).*

2. **Aplicar la configuración de WSL**
   Luego, abre una terminal (CMD o PowerShell) y ejecuta:
   `wsl --shutdown`
   (Luego Docker Desktop se reiniciará automáticamente o puedes abrirlo manualmente).

3. **Escalabilidad Inteligente en los Scripts (Automático)**
   - El archivo `escaner_encodings.py` ya está programado para **detectar la RAM libre dinámicamente** y ajustar la cantidad de _workers_ paralelos. 
   - No necesitas editar `1_escanear_fotos.bat` a menos que desees forzar límites directos de Docker. Si seguiste los pasos con `.wslconfig`, el script interno detectará que ahora tienes, por ejemplo, los 10GB de memoria, y ejecutará exactamente la cantidad máxima segura de hilos.

## Paso a paso

### Paso 1 — Escanear el lote de fotos

Este es el paso más lento. Extrae los vectores faciales (encodings) de todas las fotos y los guarda en un archivo `.pkl`.

1. Coloca todas las fotos a analizar dentro de la carpeta `fotos/` (acepta subcarpetas).
2. Da **doble clic** en el archivo:
   `1_escanear_fotos.bat`

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
3. Da **doble clic** en el archivo:
   `2_definir_objetivo.bat`

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

1. Da **doble clic** en el archivo:
   `3_buscar_objetivo.bat`

2. El script automáticamente carga los archivos de los pasos anteriores y muestra los resultados:

```
Concordancias confirmadas: 12
Revision manual (borderline): 3
Fotos unicas con esta persona: 12

Top 10 mejores coincidencias:
  1. [MUY ALTA] IMG_9076.JPG (score: 92.3, dist: 0.1454, votos: 5/5)
  2. [MUY ALTA] IMG_9084.JPG (score: 89.1, dist: 0.1467, votos: 5/5)
```

3. Se genera:
   - `personas_encontradas/resultado_busqueda.xlsx` — Excel con 5 hojas:
     - **Coincidencias**: Matches confirmados con todas las métricas
     - **Revisión Manual**: Matches borderline que fallaron 1 filtro (revisar visualmente)
     - **Fotos Únicas**: Resumen sin duplicados
     - **Lote Completo**: Todas las fotos con indicador SI/NO
     - **Configuración**: Parámetros usados (para reproducibilidad)
   - `personas_encontradas/{nombre_persona}/` — Carpeta con copias de todas las fotos donde aparece la persona

**Este paso es instantáneo (~2 segundos). Puedes ejecutarlo múltiples veces cambiando la tolerancia.**

## Estructura de carpetas

```
py_faces/
├── 0_construir_imagen.bat        # Setup inicial de Docker
├── 1_escanear_fotos.bat          # Ejecuta el paso 1 en Docker
├── 2_definir_objetivo.bat        # Ejecuta el paso 2 en Docker
├── 3_buscar_objetivo.bat         # Ejecuta el paso 3 en Docker
├── 4_escanear_videos.bat         # Ejecuta el escáner de videos
├── 5_ver_fotos_sobrantes.bat     # Separa fotos que no se le asignaron a nadie
├── Dockerfile                    # Receta de la imagen Docker
├── .dockerignore                 # Evita cargar archivos pesados al construir 
├── requirements.txt              # Dependencias de Python
├── escaner_encodings.py          # Script de Python paso 1
├── definir_objetivo.py           # Paso 2: crear perfil de persona
├── buscador_objetivo.py          # Paso 3: buscar persona en lote
├── filtrar_sobrantes.py          # Script independiente: aislar sobrantes
├── escaner_videos.py             # Script independiente para fotos + videos
├── fotos/                        # Carpeta con las fotos a analizar
├── persona_objetivo/             # Fotos de la persona a buscar (1 cara c/u)
├── output_escaner_encodings/     # Salida del paso 1
│   └── lote_encodings.pkl
├── output_comparador_caras/      # Salida del paso 2
│   └── perfil_objetivo.pkl
└── personas_encontradas/         # Salida del paso 3
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
| `MODO_NINOS` | `True` | Activa filtrado reforzado para caras de niños |
| `TOLERANCIA_BUSQUEDA` | `0.33` (niños) / `0.40` (adultos) | Distancia máxima para considerar coincidencia |
| `UMBRAL_VOTACION` | `0.75` (niños) / `0.60` (adultos) | % mínimo de muestras que deben confirmar |
| `MAX_STD_DISTANCIAS` | `0.04` (niños) / `0.06` (adultos) | Desviación estándar máxima (consistencia) |
| `MIN_SCORE_COMPUESTO` | `70` (niños) / `55` (adultos) | Score mínimo compuesto (0-100) |

### Sistema Anti-Falsos Positivos (v2)

El buscador usa **5 capas de filtrado** para eliminar falsos positivos:

1. **Distancia media** — La distancia promedio a todas las muestras del perfil debe ser ≤ tolerancia
2. **Votación por consenso** — Un porcentaje mínimo de muestras debe reconocer la cara individualmente 
3. **Consistencia (STD)** — La desviación estándar de las distancias debe ser baja (rechaza matches ambiguos)
4. **Mediana** — La mediana de distancias también debe pasar (robusto contra outliers)
5. **Score compuesto** — Combinación ponderada de todas las métricas ≥ umbral mínimo

Las caras que pasan **algunas pero no todas** las capas van a la hoja "Revisión Manual" del Excel.

## Estados de las fotos

| Estado | Significado |
|--------|-------------|
| **OK (N caras)** | Imagen leída correctamente, se detectaron N rostros |
| **SIN CARA** | Imagen leída correctamente, pero no se detectó ningún rostro |
| **ERROR** | El archivo no se pudo abrir (corrupto, formato no soportado) |

## Niveles de confianza

| Nivel | Score | Significado |
|-------|-------|-------------|
| **MUY ALTA** | ≥ 85 | Coincidencia casi segura |
| **ALTA** | 70-84 | Coincidencia confiable |
| **MEDIA** | 55-69 | Probable coincidencia, verificar manualmente |
| **BAJA** | 40-54 | Coincidencia débil, posible falso positivo |
| **MUY BAJA** | < 40 | No se incluye en resultados |

## Notas técnicas

- Los **encodings faciales** son vectores de 128 números de punto flotante que representan la "huella digital" de un rostro. Son generados por la librería `dlib` a través de `face_recognition`.
- El archivo `.pkl` (pickle) es un formato binario de Python que preserva la precisión completa de los vectores numéricos.
- El paso 1 es el cuello de botella porque cada imagen requiere detección facial (HOG/CNN) y cálculo de encoding con múltiples jitters. Los pasos 2 y 3 solo hacen operaciones matemáticas sobre vectores ya calculados.
- **Sobre Docker:** El uso de Docker con los archivos `.bat` permite "ensamblar" un pequeño entorno Linux para evitar instalar un compilador completo de C++ en tu anfitrión de Windows. El archivo `.dockerignore` juega un papel crítico para ignorar la transferencia de gigabytes de fotografías al *demonio* de Docker durante la construcción. Al momento de las búsquedas, el volumen montado del anfitrión (`-v "%~dp0:/app"`) permite la lectura/escritura de las imágenes y Excel directamente en la carpeta nativa de Windows.


## ¿Cómo funciona al nivel de tecnologías? (Explicación simplificada)

El proyecto utiliza un conjunto de herramientas para funcionar como un **detective digital** que busca personas específicas en muchas fotos o videos:

1. **Python (El cerebro):** Da las instrucciones y conecta todas las demás herramientas.
2. **Reconocimiento Facial (face_recognition y dlib):** La Inteligencia artificial (IA) que mira las fotos y se "aprende" los rasgos de la persona para buscarla.
3. **Visión artificial (OpenCV, Numpy, Pillow):** Actúan como los "ojos" súper rápidos para leer miles de fotos o fotogramas de video e identificar dónde hay un rostro.
4. **Reportes y Datos (Pandas y Excel):** Como un oficinista, guarda registro y entrega hojas de cálculo ordenadas con las fotos donde apreció la persona.
5. **Menús automatizados (Click y Archivos `.bat`):** Permiten al usuario hacer todo con dos clics, sin necesidad de programar un solo comando.
6. **Docker (La caja mágica):** Empaqueta todo este sistema en un ambiente seguro para garantizar que funcione en cualquier computadora sin instalaciones complicadas.
7. **Seguridad (Hash):** Un pequeño candado inicial que verifica las contraseñas de licencia para asegurar que solo usuarios autorizados pueden usar el sistema.

## POR AÑADIR
- flag en archivos con caras ya organizadas