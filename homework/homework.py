#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import os
import json
import gzip
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""
Función que carga un dataset en formato .zip
"""
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, compression='zip')


"""
Función que procesa los datos.

Crea la columna 'Age' a partir de la columna 'Year'.
Elimina las columnas 'Year' y 'Car_Name'.
"""
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['Age'] = 2021 - df['Year']
    df.drop(columns=['Year', 'Car_Name'], inplace=True)
    return df


"""
Función que crea el pipeline para el modelo de regresión lineal.

El pipeline debe contener las siguientes capas:
- Transforma las variables categoricas usando el método one-hot-encoding.
- Escala las variables numéricas al intervalo [0, 1].
- Selecciona las K mejores entradas.
- Ajusta un modelo de regresion lineal.
"""
def create_pipeline(x: pd.DataFrame) -> Pipeline:
    cat_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    num_features = list(set(x) - set(cat_features))

    transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(dtype="int"), cat_features),
            ("num", MinMaxScaler(), num_features),
        ],
        remainder='passthrough'
    )

    return Pipeline([
        ('tranformer', transformer),
        ('selectkbest', SelectKBest(score_func=f_classif)),
        ('estimator', LinearRegression())
    ])


"""
Función que ajusta los hiperparametros del pipeline usando validación cruzada.
"""
def create_estimator(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train, 
    cv: int = 10
):
    param_grid = {
        'selectkbest__k': [3, 4, 5, 6, 7],
        'estimator__fit_intercept': [True, False]
    }

    estimator = GridSearchCV(
        pipeline,
        param_grid=param_grid, 
        cv=cv, 
        scoring='neg_mean_absolute_error',
        verbose=2
    )
    estimator.fit(x_train, y_train)
    return estimator


"""
Función que vuelve a crear una carpeta si ya existe, o la crea si no existe.
"""
def create_folder(path: str):
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    else:
        os.mkdir(path)


"""
Función que calcula las métricas r2, error cuadratico medio y error absoluto medio.
"""
def calculate_metrics(y_true, y_pred, dataset_name):
    return {
        'type': 'metrics',
        'dataset': dataset_name,
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'mad': mean_absolute_error(y_true, y_pred)
    }


"""
Ejecuta el programa.
"""
def main():
    input_files_path = 'files/input/'
    output_models_path = 'files/models/'
    output_metrics_path = 'files/output/'

    # Carga los datasets
    train = load_dataset(os.path.join(input_files_path, 'train_data.csv.zip'))
    test = load_dataset(os.path.join(input_files_path, 'test_data.csv.zip'))

    # Paso 1 - Preprocesamiento de los datos
    train = preprocess_data(train)
    test = preprocess_data(test)

    # Paso 2 - Divide los datasets en x_train, y_train, x_test, y_test
    x_train, y_train = train.drop(columns=['Present_Price']), train['Present_Price']
    x_test, y_test = test.drop(columns=['Present_Price']), test['Present_Price']

    # Paso 3 - Crea el pipeline
    pipeline = create_pipeline(x_train)

    # Paso 4 - Ajuste de los hiperparametros
    estimator = create_estimator(pipeline, x_train, y_train)

    # Paso 5 - Guardado del modelo
    create_folder(output_models_path)
    with gzip.open(os.path.join(output_models_path, 'model.pkl.gz'), 'wb') as f:
        pickle.dump(estimator, f)

    # Paso 6 - Calculo de las metricas
    create_folder(output_metrics_path)
    open(os.path.join(output_metrics_path, 'metrics.json'), 'x').close()
    
    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)
    
    metrics = [
        calculate_metrics(y_train, y_train_pred, 'train'),
        calculate_metrics(y_test, y_test_pred, 'test')
    ]

    with open(os.path.join(output_metrics_path, 'metrics.json'), 'w') as f:
        for metric in metrics:
            json.dump(metric, f)
            f.write('\n')


if __name__ ==  '__main__':
    main()