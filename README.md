# Jordan_vs_Lebron_Prediction

### Pràctica Kaggle APC UAB 2022-23
#### Nom: Jonathan Rojas Granda 
#### DATASET: Lebron vs Jordan
#### URL: [Lebron-vs-Jordan Kaggle](https://www.kaggle.com/datasets/edgarhuichen/nba-players-career-game-log)
## Resum

### Objectius del dataset

## Experiments

### Preprocessat

### Model
#### SIN ESTANDARIZAR LOS DATOS Y MODELOS BÁSICOS
| Model | Accuracy | Recall | Precision | F1-Score | Temps |
| -- | -- | -- | -- | -- | -- |
| Logistic Regression | 0.823 |  0.822 |  0.821 | 0.822 | 0.204s |
| SVC | 0.948 | 0.947 | 0.948 | 0.948 | 0.155s |
| KNN | 0.867 |  0.866 | 0.865 | 0.866 | 0.003s |
| RandomForestClassifier | 1.0 | 1.0 | 1.0 | 1.0 | 0.122s |

#### ESTANDARIZANDO LOS DATOS Y MODELOS BÁSICOS
| Model | Accuracy | Recall | Precision | F1-Score | Temps |
| -- | -- | -- | -- | -- | -- |
| Logistic Regression | 0.821 | 0.820 | 0.820 | 0.820 | 0.014s |
| SVC | 0.998 | 0.998 | 0.998 | 0.998 | 0.065s |
| KNN | 0.990 | 0.990 | 0.990 | 0.990 | 0.001s |
| RandomForestClassifier | 1.0 | 1.0 | 1.0 | 1.0 | 0.107s |

#### NORMALIZANDO LOS DATOS Y MODELOS BÁSICOS
| Model | Accuracy | Recall | Precision | F1-Score | Temps |
| -- | -- | -- | -- | -- | -- |
| Logistic Regression | 0.785 | 0.779 | 0.786 | 0.781 | 0.151s |
| SVC | 0.965 | 0.963 | 0.966 | 0.964 | 0.157s |
| KNN | 0.891 | 0.888 | 0.892 | 0.889 | 100ms |
| RandomForestClassifier | 0.987 | 0.988 | 0.987 | 0.987 | 0.151s |

#### DATOS ESTANDARIZADOS Y MODELOS CON HYPERPARAMETROS
| Model | Hiperparametres | Accuracy | Recall | Precision | F1-Score | Temps |
| -- | -- | -- | -- | -- | -- | -- |
| Logistic Regression | 'C': 1, 'penalty': 'l1', 'solver': 'saga' | 0.82 | 0.82 | 0.82 | 0.82 | 1.08s |
| SVC | 'C': 1, 'degree': 3, 'kernel': 'rbf' | 1.0 | 1.0 | 1.0 | 1.0 | 79.66s |
| KNN | 'n_neighbors': 2, 'p': 2, 'weights': 'uniform' | 0.99 | 0.99 | 0.99 | 0.99 | 0.937s |
| RandomForestClassifier | 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto', 'n_estimators': 500 | 1.0 | 1.0 | 1.0 | 1.0 | 155.551s |

#### CROSS-VALIDATION-SCORE CON LOS HYPERPARAMETROS PARA CADA MODELO
| Model | Hiperparametres | Score |
| -- | -- | -- |
| Logistic Regression | 'C': 1, 'penalty': 'l1', 'solver': 'saga' | 0.843 |
| SVC | 'C': 1, 'degree': 3, 'kernel': 'rbf' | 1.0 |
| KNN | 'n_neighbors': 2, 'p': 2, 'weights': 'uniform' | 0.988 |
| RandomForestClassifier | 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto', 'n_estimators': 500 | 1.0 |

## Guide
Per tal de fer una prova, es pot fer servir amb la següent comanda
``` python3 demo/demo.py --input here ```

## Conclusions


## Idees per treballar en un futur
