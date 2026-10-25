import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")

RUTA_ENTRADA = r"datos/MPI_subnational_refined.csv"
CARPETA_SALIDA = "resultados"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Nombres cortos para trabajar con las variables numéricas
VARIABLES = ["MPI Regional", "Headcount Ratio Regional", "Intensity of deprivation Regional"]

# Corrección de nombres: en el CSV original las letras con tilde fueron reemplazadas por "ð"
# (la información se perdió en la fuente, por lo que se corrige manualmente)
CORRECCIONES = {
    "Ouðmð": "Ouémé", "Potosð": "Potosí", "Amapð": "Amapá", "Cearð": "Ceará",
    "Espðrito Santo": "Espírito Santo", "Goiðs": "Goiás", "Maranhðo": "Maranhão",
    "Paranð": "Paraná", "Paraðba": "Paraíba", "Parð": "Pará", "Piauð": "Piauí",
    "Rondðnia": "Rondônia", "Sðo Paulo": "São Paulo", "Mambðrð-Kadði": "Mambéré-Kadéï",
    "Nana-Mambðrð": "Nana-Mambéré", "Sangha-Mbaðrð": "Sangha-Mbaéré", "Ouham-Pendð": "Ouham-Pendé",
    "Kðmo": "Kémo", "Nana-Grðbizi": "Nana-Grébizi", "Centre (sans Yaoundð)": "Centre (sans Yaoundé)",
    "Extrðme-Nord": "Extrême-Nord", "Yaoundð": "Yaoundé", "ðquateur": "Équateur",
    "Kasað-Occidental": "Kasaï-Occidental", "Kasað-Oriental": "Kasaï-Oriental", "Lðkoumou": "Lékoumou",
    "Cauca Y Nariðo Sin Litoral": "Cauca Y Nariño Sin Litoral", "Mwali (Mohðli)": "Mwali (Mohéli)",
    "Haut-Ogoouð": "Haut-Ogooué", "Moyen-Ogoouð": "Moyen-Ogooué", "Ngounið": "Ngounié",
    "Ogoouð Maritime (sans Port-Gentil)": "Ogooué Maritime (sans Port-Gentil)",
    "Ogoouð-Ivindo": "Ogooué-Ivindo", "Ogoouð-Lolo": "Ogooué-Lolo", "Woleu-Nðtem": "Woleu-Ntem",
    "Bokð": "Boké", "Labð": "Labé", "N'Zðrðkorð": "N'Zérékoré", "Atlðntida": "Atlántida",
    "Colðn": "Colón", "Copðn": "Copán", "Cortðs": "Cortés", "El Paraðso": "El Paraíso",
    "Francisco Morazðn": "Francisco Morazán", "Intibucð": "Intibucá", "Islas de la Bahða": "Islas de la Bahía",
    "Santa Bðrbara": "Santa Bárbara", "Aire Mðtropolitaine/Reste-Ouest": "Aire Métropolitaine/Reste-Ouest",
    "Rðgion Oriental": "Région Oriental", "Rabat-Salð-Zemmour-Zair": "Rabat-Salé-Zemmour-Zair",
    "Sðgou": "Ségou", "Zambðzia": "Zambézia", "Maputo Provðncia": "Maputo Província",
    "ðncash": "Áncash", "Apurðmac": "Apurímac", "Huðnuco": "Huánuco", "Junðn": "Junín",
    "San Martðn": "San Martín", "GuðRa": "Guéra", "N'DjamðNa": "N'Djaména",
    "Ouaddað": "Ouaddaï", "Tandjilð": "Tandjilé",
}


# ============================================================
# FASE 3: TRANSFORMACIÓN
# ============================================================
def transformar():
    # Cargamos el archivo limpio. Se prueba utf-8 y, si falla, latin-1
    try:
        datos = pd.read_csv(RUTA_ENTRADA, encoding="utf-8")
    except UnicodeDecodeError:
        datos = pd.read_csv(RUTA_ENTRADA, encoding="latin-1")

    # El archivo refinado guardó el índice como columna "Unnamed: 0", la eliminamos
    datos = datos.drop(columns=[c for c in datos.columns if c.startswith("Unnamed")])

    # Revisamos si hay caracteres incorrectos en los nombres (problema de codificación)
    con_error = datos["Sub-national region"].str.contains("ð", na=False)
    print(f"Registros con caracteres incorrectos en la región (antes): {con_error.sum()}")
    datos["Sub-national region"] = datos["Sub-national region"].replace(CORRECCIONES)
    pendientes = datos.loc[datos["Sub-national region"].str.contains("ð", na=False), "Sub-national region"]
    print(f"Registros con caracteres incorrectos (después): {len(pendientes)}")

    print("\nDimensiones del dataset:", datos.shape)
    print(datos.dtypes)

    # 1. Discretización: se crean categorías de pobreza con terciles del MPI Regional
    datos["Nivel de pobreza"] = pd.qcut(
        datos["MPI Regional"], q=3, labels=["Bajo", "Medio", "Alto"]
    )
    print("\nLímites de cada nivel (terciles del MPI Regional):")
    print(datos.groupby("Nivel de pobreza", observed=True)["MPI Regional"].agg(["min", "max", "count"]))

    # 2. Reducción: se conservan las variables necesarias para la minería
    # (MPI National se mantiene solo para la comparación nacional vs. regional)
    datos["Brecha MPI (Regional - Nacional)"] = datos["MPI Regional"] - datos["MPI National"]

    # 3. Normalización: el Headcount va de 0 a 100 y el MPI de 0 a 1, por lo que se estandarizan
    escalador = StandardScaler()
    escaladas = pd.DataFrame(
        escalador.fit_transform(datos[VARIABLES]),
        columns=[v + " (z)" for v in VARIABLES],
        index=datos.index,
    )
    datos = pd.concat([datos, escaladas], axis=1)

    datos.to_csv(os.path.join(CARPETA_SALIDA, "MPI_subnational_transformado.csv"), index=False)
    return datos


# ============================================================
# FASE 4: MINERÍA DE DATOS
# ============================================================
def analisis_descriptivo(datos):
    print("\n===== ANÁLISIS DESCRIPTIVO =====")
    print(datos[VARIABLES].describe().round(3))

    # Promedios por región del mundo
    por_region = datos.groupby("World region")[VARIABLES].mean().round(3)
    por_region["Registros"] = datos.groupby("World region").size()
    print("\nPromedios por región del mundo:")
    print(por_region.sort_values("MPI Regional", ascending=False))
    por_region.to_csv(os.path.join(CARPETA_SALIDA, "promedios_region_mundo.csv"))

    # Ranking de las 10 regiones subnacionales con mayor MPI
    top10 = datos.nlargest(10, "MPI Regional")[
        ["Country", "Sub-national region", "MPI Regional", "Headcount Ratio Regional",
         "Intensity of deprivation Regional"]
    ]
    print("\nTop 10 regiones con mayor pobreza multidimensional:")
    print(top10.to_string(index=False))
    top10.to_csv(os.path.join(CARPETA_SALIDA, "top10_regiones.csv"), index=False)

    # Países con mayor desigualdad interna (diferencia entre su región más y menos pobre)
    desigualdad = datos.groupby("Country")["MPI Regional"].agg(["min", "max", "count"])
    desigualdad["Rango"] = desigualdad["max"] - desigualdad["min"]
    print("\nTop 10 países con mayor desigualdad interna (rango del MPI Regional):")
    print(desigualdad.sort_values("Rango", ascending=False).head(10).round(3))

    # Gráfico 1: histogramas
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
    for eje, var in zip(ejes, VARIABLES):
        sns.histplot(datos[var], bins=30, kde=True, ax=eje)
        eje.set_title(var)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "01_histogramas.png"), dpi=150)
    plt.close()

    # Gráfico 2: boxplot del MPI Regional por región del mundo
    plt.figure(figsize=(10, 5))
    orden = por_region.sort_values("MPI Regional", ascending=False).index
    sns.boxplot(data=datos, y="World region", x="MPI Regional", order=orden)
    plt.title("MPI Regional por región del mundo")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "02_boxplot_region_mundo.png"), dpi=150)
    plt.close()


def correlacion(datos):
    print("\n===== CORRELACIÓN =====")
    matriz = datos[VARIABLES].corr().round(3)
    print(matriz)
    matriz.to_csv(os.path.join(CARPETA_SALIDA, "matriz_correlacion.csv"))

    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, cmap="Blues", vmin=0, vmax=1)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "03_correlacion.png"), dpi=150)
    plt.close()

    # Dispersión MPI vs Headcount, coloreado por intensidad
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(datos["Headcount Ratio Regional"], datos["MPI Regional"],
                     c=datos["Intensity of deprivation Regional"], cmap="viridis", alpha=0.7)
    plt.colorbar(sc, label="Intensidad de privación")
    plt.xlabel("Headcount Ratio Regional (%)")
    plt.ylabel("MPI Regional")
    plt.title("MPI Regional vs. Headcount Ratio Regional")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "04_dispersion_mpi_headcount.png"), dpi=150)
    plt.close()


def clustering(datos, k_final=3):
    print("\n===== CLUSTERING (K-MEANS) =====")
    columnas_z = [v + " (z)" for v in VARIABLES]
    X = datos[columnas_z]

    # Método del codo e índice de silueta para elegir k
    inercias, siluetas, rango_k = [], [], range(2, 9)
    for k in rango_k:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inercias.append(modelo.inertia_)
        siluetas.append(silhouette_score(X, modelo.labels_))
        print(f"k = {k} | inercia = {modelo.inertia_:.1f} | silueta = {siluetas[-1]:.3f}")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    ejes[0].plot(list(rango_k), inercias, marker="o")
    ejes[0].set_title("Método del codo")
    ejes[0].set_xlabel("k")
    ejes[0].set_ylabel("Inercia")
    ejes[1].plot(list(rango_k), siluetas, marker="o", color="darkorange")
    ejes[1].set_title("Índice de silueta")
    ejes[1].set_xlabel("k")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "05_codo_silueta.png"), dpi=150)
    plt.close()

    # Modelo final
    modelo = KMeans(n_clusters=k_final, random_state=42, n_init=10).fit(X)
    datos["Cluster"] = modelo.labels_

    # Se renombran los clusters según su MPI promedio (0 = menor pobreza)
    orden = datos.groupby("Cluster")["MPI Regional"].mean().sort_values().index
    nombres = {cl: n for cl, n in zip(orden, ["Pobreza baja", "Pobreza media", "Pobreza alta"][:k_final])}
    datos["Grupo (K-Means)"] = datos["Cluster"].map(nombres)

    perfil = datos.groupby("Grupo (K-Means)")[VARIABLES].mean().round(3)
    perfil["Registros"] = datos.groupby("Grupo (K-Means)").size()
    print(f"\nPerfil de los {k_final} clusters:")
    print(perfil.sort_values("MPI Regional"))
    perfil.to_csv(os.path.join(CARPETA_SALIDA, "perfil_clusters.csv"))

    # Composición de cada cluster por región del mundo
    composicion = pd.crosstab(datos["World region"], datos["Grupo (K-Means)"])
    print("\nRegiones del mundo por cluster:")
    print(composicion)
    composicion.to_csv(os.path.join(CARPETA_SALIDA, "clusters_por_region_mundo.csv"))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=datos, x="Headcount Ratio Regional", y="Intensity of deprivation Regional",
                    hue="Grupo (K-Means)", hue_order=["Pobreza baja", "Pobreza media", "Pobreza alta"][:k_final],
                    palette="viridis", alpha=0.8)
    plt.title("Clusters de regiones según incidencia e intensidad")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "06_clusters.png"), dpi=150)
    plt.close()

    datos.to_csv(os.path.join(CARPETA_SALIDA, "MPI_subnational_clusters.csv"), index=False)


def clasificacion(datos):
    print("\n===== CLASIFICACIÓN (ÁRBOL DE DECISIÓN) =====")
    # Se predice el nivel de pobreza usando solo Headcount e Intensidad.
    # No se incluye el MPI Regional porque con él se construyó la etiqueta (fuga de información).
    predictores = ["Headcount Ratio Regional", "Intensity of deprivation Regional"]
    X = datos[predictores]
    y = datos["Nivel de pobreza"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    arbol = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    pred = arbol.predict(X_test)

    print(f"Exactitud (accuracy): {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred))
    clases = ["Bajo", "Medio", "Alto"]
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(confusion_matrix(y_test, pred, labels=clases), index=clases, columns=clases))

    plt.figure(figsize=(14, 7))
    plot_tree(arbol, feature_names=predictores, class_names=list(arbol.classes_),
              filled=True, rounded=True, fontsize=9)
    plt.title("Árbol de decisión para el nivel de pobreza")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "07_arbol_decision.png"), dpi=150)
    plt.close()


if __name__ == '__main__':
    # Fase 3: transformación
    datos_transformados = transformar()
    # Fase 4: minería de datos
    analisis_descriptivo(datos_transformados)
    correlacion(datos_transformados)
    clustering(datos_transformados)
    clasificacion(datos_transformados)
    print(f"\nProceso terminado. Resultados guardados en la carpeta '{CARPETA_SALIDA}'.")