import pandas as pd

# Cargar el documento csv
datos_pobreza = pd.read_csv(r"datos/MPI_subnational.csv")
pd.set_option('display.max_columns', None)

def imprimir_informacion():
    # Verificamos los tipos de datos con los que trabajamos
    print(datos_pobreza.info())
    # Identificar si existe algun dato missing en alguna columna
    print(datos_pobreza.isna().sum())
    # Identificamos de cuál registro se trata el dato faltante
    print(datos_pobreza[datos_pobreza["Intensity of deprivation Regional"].isna()])

def limpiar_csv():
    # Eliminamos el registro con dato missing
    nuevo_registro_pobreza = datos_pobreza.dropna(subset=["Intensity of deprivation Regional"])
    # Obtenermos un nuevo archivo con los datos limpios
    nuevo_registro_pobreza.to_csv(r"datos/MPI_subnational_refined.csv")

if __name__ == '__main__':
    # La variable "Intensity of deprivation Regional" contiene un dato faltante.
    imprimir_informacion()
    # Eliminar el registro con datos faltantes y obtener un nuevo dataset
    limpiar_csv()