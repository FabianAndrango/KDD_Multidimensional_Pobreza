# Proceso KDD: Análisis de Pobreza Multidimensional Subnacional

Este proyecto implementa el ciclo completo de **Descubrimiento de Conocimiento en Bases de Datos (KDD - Knowledge Discovery in Databases)** aplicado al **Índice de Pobreza Multidimensional (MPI - Multidimensional Poverty Index)** a nivel subnacional / regional en diversos países del mundo.

---

## 📋 Estructura del Proyecto

```text
KDD_Multidimensional_Pobreza/
│
├── datos/
│   ├── MPI_subnational.csv               # Dataset original (984 registros)
│   └── MPI_subnational_refined.csv       # Dataset limpio tras eliminar valores nulos (983 registros)
│
├── resultados/                           # Artefactos generados automáticamente
│   ├── 01_histogramas.png                # Distribución de las variables principales
│   ├── 02_boxplot_region_mundo.png       # Boxplot de MPI por macro-región mundial
│   ├── 03_correlacion.png                # Heatmap de correlación de Pearson
│   ├── 04_dispersion_mpi_headcount.png   # Dispersión MPI vs Incidencia e Intensidad
│   ├── 05_codo_silueta.png               # Curvas de inercia y silueta para K-Means
│   ├── 06_clusters.png                   # Dispersión de los 3 clusters generados
│   ├── 07_arbol_decision.png             # Visualización del árbol de decisión (profundidad 3)
│   ├── MPI_subnational_transformado.csv  # Dataset con discretización y estandarización Z
│   ├── MPI_subnational_clusters.csv      # Dataset final con asignación de clusters
│   ├── clusters_por_region_mundo.csv     # Distribución de clusters por región mundial
│   ├── matriz_correlacion.csv            # Matriz numérica de correlación
│   ├── perfil_clusters.csv               # Promedios y tamaño de cada cluster
│   ├── promedios_region_mundo.csv        # Promedios de variables por región
│   └── top10_regiones.csv                # Top 10 regiones con mayor pobreza
│
├── limpieza_de_datos.py                  # Script de las Fases 1 y 2 (Selección y Limpieza)
├── transformacion_y_mineria.py           # Script de las Fases 3, 4 y 5 (Transformación, Modelado y Evaluación)
├── requirements.txt                      # Lista de dependencias del entorno Python
└── README.md                             # Documentación del proyecto
```

---

## ⚙️ Requisitos Previos

- **Python:** Versión 3.10 o superior (compatible con Python 3.11, 3.12 y 3.13).
- **pip:** Gestor de paquetes de Python actualizado.

---

## 🚀 Instalación y Configuración

### 1. Clonar o acceder al repositorio
Abre una terminal en la carpeta principal del proyecto:
```bash
cd KDD_Multidimensional_Pobreza
```

### 2. Crear el entorno virtual (si no existe)
```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

- **En PowerShell (Windows):**
  > ⚠️ **Nota sobre permisos en PowerShell:** Si te aparece el error `la ejecución de scripts está deshabilitada en este sistema`, ejecuta primero:
  > ```powershell
  > Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  > ```
  Luego activa el entorno:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- **En Símbolo del Sistema / CMD (Windows):**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **En Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Instalar las dependencias
Con el entorno virtual activado, instala las librerías necesarias:
```bash
pip install -r requirements.txt
```

---

## 🏃 Modo de Uso / Ejecución

El proceso KDD está estructurado en dos scripts ejecutables en orden secuencial:

### Paso 1: Limpieza y Preprocesamiento Inicial
Ejecuta el script que inspecciona el dataset, identifica registros con valores ausentes (`NaN`) y genera el archivo `datos/MPI_subnational_refined.csv`:

```bash
python limpieza_de_datos.py
```

### Paso 2: Transformación, Minería y Evaluación
Ejecuta el pipeline completo de transformación, análisis exploratorio, clustering y clasificación supervisada:

```bash
python transformacion_y_mineria.py
```

> Al finalizar, todos los gráficos (`.png`) y tablas procesadas (`.csv`) se guardarán automáticamente dentro del directorio `resultados/`.

---

## 🔬 Fases del Proceso KDD Implementadas

### 1. Selección y 2. Limpieza de Datos (`limpieza_de_datos.py`)
- Se identificó un único valor faltante en la variable `Intensity of deprivation Regional` correspondiente al registro de Callao, Perú.
- Se aplicó eliminación por lista (`dropna`), conservando 983 observaciones válidas.

### 3. Transformación (`transformar()`)
- **Corrección léxica/codificación:** Se solucionó la pérdida de tildes y caracteres especiales codificados erróneamente como `ð` en 62 regiones subnacionales mediante un diccionario manual de mapeo fonético/geográfico.
- **Discretización:** Clasificación de la variable objetivo `MPI Regional` en terciles:
  - **Bajo:** [0.000, 0.082]
  - **Medio:** [0.083, 0.280]
  - **Alto:** [0.283, 0.744]
- **Ingeniería de Características:** Cálculo de la `Brecha MPI (Regional - Nacional)` para medir disparidades territoriales internas.
- **Normalización:** Estandarización con `StandardScaler` (puntajes Z) para evitar sesgos de escala entre porcentajes (0-100) e índices (0-1).

### 4. Minería de Datos (`transformacion_y_mineria.py`)
1. **Análisis Descriptivo y Correlación:**
   - Cálculo de promedios ponderados y rankings de desigualdad territorial (destacando países como Nigeria, Etiopía y Uganda).
   - Matriz de correlación de Pearson demostrando fuerte asociación lineal entre incidencia e intensidad ($r > 0.90$).
2. **Clustering No Supervisado (K-Means):**
   - Selección de $k=3$ clusters fundamentada en el análisis combinado del Método del Codo y el Coeficiente de Silueta ($0.546$).
   - Segmentación en perfiles: *Pobreza baja* (501 regiones), *Pobreza media* (288 regiones) y *Pobreza alta* (194 regiones).
3. **Clasificación Supervisada (Árbol de Decisión):**
   - Entrenamiento de un modelo `DecisionTreeClassifier` (profundidad máxima 3) utilizando únicamente `Headcount Ratio Regional` e `Intensity of deprivation Regional` como predictores (evitando fuga de datos con respecto al índice compuesto).
   - Partición estratificada 70% entrenamiento y 30% prueba.
   - Rendimiento sobresaliente con **Accuracy de ~98.6%**, evaluado mediante matriz de confusión y reporte de clasificación completo.

---

## 📦 Librerías Principales

| Librería | Propósito |
| :--- | :--- |
| **pandas** | Manipulación y estructuración de datos tabulares |
| **numpy** | Operaciones numéricas y cálculo de matrices |
| **matplotlib** | Generación de gráficos y guardado de figuras |
| **seaborn** | Visualizaciones estadísticas avanzadas y mapas de calor |
| **scikit-learn** | Algoritmos de Machine Learning (KMeans, DecisionTree, StandardScaler, métricas) |
| **scipy** | Funciones matemáticas y soporte estadístico |
