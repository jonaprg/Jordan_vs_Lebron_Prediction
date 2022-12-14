# Jordan_vs_Lebron_Prediction

### Práctica Kaggle APC UAB 2022-23
#### Nombre: Jonathan Rojas Granda 
#### DATASET: Lebron vs Jordan
#### URL: [Lebron-vs-Jordan Kaggle](https://www.kaggle.com/datasets/edgarhuichen/nba-players-career-game-log)

## Resumen
Este conjunto de datos contiene los datos sobre la carrera de Jordan y Lebron de la NBA. Los partidos jugados son similares, por lo que es una buena muestra para comparar a dos jugadores. En este dataset, hay 26 atributos, el cuales son estadísticas sobre el partido.

### Objectivo del dataset
El objetivo de este dataset, será predecir el jugador, Lebron o Jordan, dependiendo de sus características de cada partido.

## Experimentos
En este notebook se implementará diferentes modelos de clasificación. Los cuales son:
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### Preprocesado
En este conjunto de datos se han procesado los datos, y sé también se han categorizado atributos para poder utilizarlos para hacer nuestra predicción.
Primero de todo se han categorizado dos atributos que son Team y Opp, utilizando el procesado LabelEncoder. También se ha estandarizado y normalizado estos datos para comprobar si es mejor ajustar nuestros datos.

### Modelos
En este apartado mostraremos las métricas sobre los modelos empleados. Para ello se ha creado cinco tablas para describir como han sido trabajadas estos modelo.
- Clasificación sin hacer el preprocesamiento estandarizado
- Clasificación haciendo estandarización de los datos
- Clasificación haciendo normalización de los datos
- Clasificación utilizando los hiperparámetros y con los datos estandarizados
- Cross-validation score con los hiperparámetros

Todos estos modelos han sido probados y ejecutados con un 70% para los datos de train, y un 30% para los datos de test.

#### SIN ESTANDARIZAR LOS DATOS Y MODELOS BÁSICOS
- El modelo con mejor F1-score es el modelo RandomForestClassifier con un puntaje de 1.0

| Model | Accuracy | Recall | Precision | F1-Score | Temps |
| -- | -- | -- | -- | -- | -- |
| Logistic Regression | 0.823 |  0.822 |  0.821 | 0.822 | 0.204s |
| SVC | 0.948 | 0.947 | 0.948 | 0.948 | 0.155s |
| KNN | 0.867 |  0.866 | 0.865 | 0.866 | 0.003s |
| RandomForestClassifier | 1.0 | 1.0 | 1.0 | 1.0 | 0.122s |

#### ESTANDARIZANDO LOS DATOS Y MODELOS BÁSICOS
- El modelo con mejor F1-score és el modelo RandomForestClassifier con un puntaje de 1.0

| Model | Accuracy | Recall | Precision | F1-Score | Temps |
| -- | -- | -- | -- | -- | -- |
| Logistic Regression | 0.821 | 0.820 | 0.820 | 0.820 | 0.014s |
| SVC | 0.998 | 0.998 | 0.998 | 0.998 | 0.065s |
| KNN | 0.990 | 0.990 | 0.990 | 0.990 | 0.001s |
| RandomForestClassifier | 1.0 | 1.0 | 1.0 | 1.0 | 0.107s |

#### NORMALIZANDO LOS DATOS Y MODELOS BÁSICOS
- El modelo con mejor F1-score es el modelo RandomForestClassifier con un puntaje de 0.98

| Model | Accuracy | Recall | Precision | F1-Score | Temps |
| -- | -- | -- | -- | -- | -- |
| Logistic Regression | 0.785 | 0.779 | 0.786 | 0.781 | 0.151s |
| SVC | 0.965 | 0.963 | 0.966 | 0.964 | 0.157s |
| KNN | 0.891 | 0.888 | 0.892 | 0.889 | 100ms |
| RandomForestClassifier | 0.987 | 0.988 | 0.987 | 0.987 | 0.151s |

#### DATOS ESTANDARIZADOS Y MODELOS CON HYPERPARAMETROS
- Los modelos con mejores F1-score son el modelo RandomForestClassifier y SVC con un puntaje de 1.0

| Model | Hiperparametres | Accuracy | Recall | Precision | F1-Score | Temps |
| -- | -- | -- | -- | -- | -- | -- |
| Logistic Regression | 'C': 1, 'penalty': 'l1', 'solver': 'saga' | 0.82 | 0.82 | 0.82 | 0.82 | 1.08s |
| SVC | 'C': 1, 'degree': 3, 'kernel': 'rbf' | 1.0 | 1.0 | 1.0 | 1.0 | 79.66s |
| KNN | 'n_neighbors': 2, 'p': 2, 'weights': 'uniform' | 0.99 | 0.99 | 0.99 | 0.99 | 0.937s |
| RandomForestClassifier | 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto', 'n_estimators': 500 | 1.0 | 1.0 | 1.0 | 1.0 | 155.551s |

#### CROSS-VALIDATION-SCORE CON LOS HYPERPARAMETROS PARA CADA MODELO
- Los modelos con mejores score son el modelo RandomForestClassifier y SVC con un puntaje de 1.0

| Model | Hiperparametres | Score |
| -- | -- | -- |
| Logistic Regression | 'C': 1, 'penalty': 'l1', 'solver': 'saga' | 0.843 |
| SVC | 'C': 1, 'degree': 3, 'kernel': 'rbf' | 1.0 |
| KNN | 'n_neighbors': 2, 'p': 2, 'weights': 'uniform' | 0.988 |
| RandomForestClassifier | 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto', 'n_estimators': 500 | 1.0 |

## Guide
Per tal de fer una prova, es pot fer servir amb la següent comanda

``` $ git clone https://github.com/migueldemollet/Activity-Recognition.git ```

Abrir el archivo Lebron-vs-JordanPreditcion.ipynb en un IDE como Visual Studio Code o pycharm.

Executar el código y modificarlo.

## Conclusiones

Se puede decir que estos dos jugadores, Lebron y Jordan, se les puede diferenciar por las estadísticas que obtenemos de NBA. Esto quiere decir que son jugadores excepcionales, ya que con tan pocos datos sobre ellos. Podemos llegar a tener una buena predicción.

También cabe destacar que haciendo diferentes pruebas sobre la base de datos separadas y prediciendo otros atributos relevantes como puede ser el equipo o el resultado. No obtenemos estos resultados mostrados anteriormente. Para ello se han juntado estos datos y se ha predicho el jugador en cuestión. También me ha parecido más relevante predecir el jugador dependiendo de las estadísticas de cada uno.

Los datos que hemos obtenido los intervalos no eran muy dispersos y eso ha hecho, como se ha visto, que sin normalizar o estandarizar estos datos, se han obtenido un resultado aceptable. Pero estandarizando estos datos y buscando los mejores hiperparámetros de cada modelo, hemos obtenido que nuestro mejor modelo clasificador, sea el SVC con el kernel rbf y el modelo RandomForestClassifier.

También, por otro lado, vemos, que con el modelo Logistic Regression y KNN vemos que utilizando su modelo básico obtenemos una mejor puntuación que normalizando sus datos, pero no estandarizando estos mismos.

## Idees per treballar en un futur

Para nuestro conjunto, si trabajamos por separado sería interesante si te gusta el baloncesto predecir los logros restantes de la carrera de Lebron o Jordan utilizando las estadísticas que nos proporciona la NBA.

## Licencia
El proyecto se ha desarrollado con la licencia del Jupyter Notebook 6.5.2
