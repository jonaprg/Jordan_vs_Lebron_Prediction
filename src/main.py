#!/usr/bin/env python
# coding: utf-8

# # LEBRON VS JORDAN PREDICTION

# - Hay un total de 26 atributos, que son los siguientes:
#     * date: fecha del partido
#     * age: edad
#     * team: equipo
#     * opp: oponente
#     * result: W:Ganado, L:Perdido
#     * mp: minutos jugado por partido
#     * fg: canasta por partido
#     * fga: canasta intentados por partido
#     * fgp: porcentaje de canasta
#     * three: tiros de 3 puntos por partido
#     * threeatt: 3 puntos de tiros intentados por partido
#     * threep: porcentaje de tiros de 3 puntos por partido
#     * ft: tiros libres por partido
#     * fta: tiros libres intentados por partido
#     * ftp: porcentaje de tiros libres por partido
#     * orb: rebotes ofensivos por partido
#     * drb: rebotes defensivos por partido
#     * trb: rebotes totales por partido
#     * ast: asistencias por partido
#     * stl: robos por partido
#     * blk: tapones por partido
#     * tov: pérdidas de balón por partido
#     * pts: puntos por partido

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc, recall_score, precision_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


import warnings
warnings.filterwarnings('ignore')
    
# import some data to play with
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

jordan_dataset = load_dataset('../data/jordan_career.csv')
lebron_dataset = load_dataset('../data/lebron_career.csv')


# ### ANÁLISIS DE LOS DATOS


def check_dataframe(dataframe):
    print('############### HEAD ###############')
    print(dataframe.head(5))
    print('############### NaN ###############')
    print(dataframe.isnull().sum())
    print('############### INFO ###############')
    print(dataframe.info())


# #### ANÁLISIS SOBRE LA BASE DE DATOS DE JORDAN


check_dataframe(jordan_dataset)


# #### ANÁLISIS SOBRE LA BASE DE DATOS DE LEBRON

check_dataframe(lebron_dataset)

jordan_dataset['player'] = 0 #JORDAN 
lebron_dataset['player'] = 1 #LEBRON
lebron_jordan = jordan_dataset.append(lebron_dataset, ignore_index=True)


check_dataframe(lebron_jordan)


# #### TRATAR LOS DATOS
# 

lebron_jordan.isnull().values.any()


lebron_jordan[lebron_jordan.duplicated()]


lebron_jordan = lebron_jordan.drop(['minus_plus'], axis=1)
lebron_jordan = lebron_jordan.drop(['date'], axis=1)
lebron_jordan['threep'].fillna(0.0, inplace=True)
lebron_jordan['ftp'].fillna(0.0, inplace=True)


lebron_jordan['result'] = lebron_jordan['result'].str[:1]
lebron_jordan["mp"] = lebron_jordan["mp"].apply(lambda x: float(str(x).replace(":",".").split(",")[0].strip()))
lebron_jordan["age"] = lebron_jordan["age"].apply(lambda x: float(str(x).replace("-",".").split(",")[0].strip()))



lebron_jordan.head()




sns.countplot(x=lebron_jordan["player"])


def plotPointsDependsOpp(dataframe):
    fig, ax = plt.subplots(figsize=(18,8))

    opp = dataframe.pivot_table(columns='player',index='opp', values='pts')
    opp.plot(ax=ax, kind='bar')

    ax.set_ylim(0, 50)
    ax.set_title("Puntos a cada oponente", fontsize=17)
    ax.legend(loc='upper right', title='Player: 0 Jordan, 1 Lebron')

    fig.autofmt_xdate()

plotPointsDependsOpp(lebron_jordan)


lebron_jordan.info()


# #### CATEGORIZAR
# 

print("####### TEAMS ############")
print(lebron_jordan['team'].unique())
print("######## OPP ############")
print(lebron_jordan['opp'].unique())


def replace_categorical(df):
    le = preprocessing.LabelEncoder()
    columns = df.columns
    for col in columns:
        if df[col].dtype == 'object':
            le.fit(df[col].astype(str))
            df[col] = le.transform(df[col].astype(str))
    return df


lebron_jordan = replace_categorical(lebron_jordan)


lebron_jordan.head()



correlacio = lebron_jordan.corr()
plt.figure(figsize=(25,25))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)


# #### PREPROCESAMIENTO DE DATOS


features = ['team', 'three', 'threeatt', 'drb', 'ast','tov']
X = lebron_jordan.loc[:, features].values
y = lebron_jordan.loc[:, 'player'].values



import time
# Matriz de confusión, y mostrarla por pantalla.
def create_confusionMatrix(true_class, preds, model_name):
    conf_matrix = confusion_matrix(y_true=true_class, y_pred=preds)
    labels = ['Class 0', 'Class 1']
    fig = plt.figure()
    model_filename = model_name.replace(' ', '_')
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d");
    plt.title('{} Confusion Matrix'.format(model_name))
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    plt.savefig("../figures/{}_confusion_matrix.png".format(model_filename))
    plt.show() 
    
#Evaluar cada modelo y mostrar las métricas.
def evaluateModel(name, model, X_train, X_test, y_train, y_test):
    initialTime = time.time()
    model.fit(X_train, y_train)
    finalTime = time.time()* initialTime
    y_pred = model.predict(X_test)
    lr_probs  = model.predict_proba(X_test)
    print("MODEL ", name )
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred, average='macro'))
    print("Precision: ", precision_score(y_test, y_pred, average='macro'))
    print('F1_score:', f1_score(y_test, y_pred, average='macro'))
    print("Time:", finalTime )

    
# Buscamos los mejores parametros de cada modelo, y también mostraremos las métricas.    
def hyperparametresForModels(name, model, params, X_train, X_test, y_train, y_test):
    gs = GridSearchCV(estimator=model, param_grid=params) #Search Hyper parametre 
    initialTime = time.time()
    gs.fit(X_train, y_train)
    print("MODEL ", name )
    print("{} Best Params: {}".format(name, gs.best_params_))
    print("{} Training score with best params: {}".format(name, gs.best_estimator_.score(X_train, y_train)))
    print("{} Test score with best params: {}".format(name, gs.best_estimator_.score(X_test, y_test)))
    y_preds = gs.best_estimator_.predict(X_test)
    print("{} prediction metrics: \n{}".format(name, classification_report(y_true=y_test, y_pred=y_preds)))
    scores = cross_val_score(model, X_train, y_train, cv=6, scoring='accuracy')
    print("Cross-validation scores:", scores)
    print("Mean:", scores.mean())
    create_confusionMatrix(y_test, y_preds, name)
    finalTime = time.time()* initialTime
    print("Time:", finalTime)




def logisticRegression(X_train, X_test, y_train, y_test, hyperparam=False):
    if hyperparam:
        
        logisticRegression = LogisticRegression(fit_intercept=True, tol=0.001)

        lr_params = {
            'C': [0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['lbfgs', 'liblinear', 'newton*cg', 'newton*cholesky', 'sag', 'saga']
        }
        hyperparametresForModels('Logistic Regression Hyper', logisticRegression, lr_params, X_train, X_test, y_train, y_test) 
    else:
        lg = LogisticRegression(fit_intercept=True, tol=0.001)
        evaluateModel("Logistic regression", lg, X_train, X_test, y_train, y_test)
        




def svc(X_train, X_test, y_train, y_test, hyperparam=False):
    if hyperparam:
        
        svc_kernel = SVC(probability=True, max_iter = 100000)

        svc_params = {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5]
        }

        hyperparametresForModels('SVC Hyper', svc_kernel, svc_params, X_train, X_test, y_train, y_test)
    else:
        model = SVC(probability=True, max_iter = 100000)
        evaluateModel("SVC", model, X_train, X_test, y_train, y_test)
        




def knn(X_train, X_test, y_train, y_test, hyperparam=False):
    if hyperparam:
        knn = KNeighborsClassifier()
        knn_params = {
            'n_neighbors': [1, 2, 5, 10, 20, 40],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        hyperparametresForModels('KNN Hyper ', knn,knn_params, X_train, X_test, y_train, y_test)
    else:
        model = KNeighborsClassifier()
        evaluateModel("KNN", model, X_train, X_test, y_train, y_test)
        




def rfc(X_train, X_test, y_train, y_test, hyperparam=False):
    if hyperparam:
        rfc = RandomForestClassifier()

        rfc_params = {
            'n_estimators': [100, 200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8, 9],
            'criterion': ['gini', 'entropy']
        }
        hyperparametresForModels('Random Forest Classifier', rfc,rfc_params, X_train, X_test, y_train, y_test)
    else:
        model = RandomForestClassifier()
        evaluateModel("RFC", model, X_train, X_test, y_train, y_test)


# ### MODELOS Y RESULTADOS

# #### DATOS SIN ESTANDARIZAR Y CLASIFICACIÓN CON MODELOS BÁSICOS



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=40)




logisticRegression(X_train, X_test, y_train, y_test)




svc(X_train, X_test, y_train, y_test)




knn(X_train, X_test, y_train, y_test)




rfc(X_train, X_test, y_train, y_test)


# #### DATOS ESTANDARIZADOS Y CLASIFICACIÓN CON MODELOS BÁSICOS



def standarize(X):
    return StandardScaler().fit(X).transform(X)

X_standarize = standarize(X)




x_t, x_v, y_t, y_v = train_test_split(X_standarize, y, train_size=0.7, random_state=40)




logisticRegression(x_t, x_v, y_t, y_v)




svc(x_t, x_v, y_t, y_v)




knn(x_t, x_v, y_t, y_v)





rfc(x_t, x_v, y_t, y_v)


#  ##### DATOS NORMALIZADOS Y CLASIFICACIÓN CON MODELOS BÁSICOS



from sklearn.preprocessing import Normalizer
def normalize(X):
    return Normalizer().fit(X).transform(X)

X_normalize = normalize(X)




x_t, x_v, y_t, y_v = train_test_split(X_normalize, y, train_size=0.7, random_state=40)




logisticRegression(x_t, x_v, y_t, y_v)




svc(x_t, x_v, y_t, y_v)




knn(x_t, x_v, y_t, y_v)




rfc(x_t, x_v, y_t, y_v)


# ##### GRID SEARCH - HIPERPARÁMETROS
# 



x_t, x_v, y_t, y_v = train_test_split(X_standarize, y, train_size=0.7, random_state=40)




logisticRegression(x_t, x_v, y_t, y_v, True)




svc(x_t, x_v, y_t, y_v, True)




knn(x_t, x_v, y_t, y_v, True)




rfc(x_t, x_v, y_t, y_v, True)


# #### ROC CURVE




from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as pyplot

X_train, X_test, y_train, y_test = train_test_split(X_standarize, y, train_size=0.7, random_state=40)
lr=LogisticRegression(C= 1, penalty= 'l1', solver= 'saga', fit_intercept=True, tol=0.001)
svm=SVC(probability=True, max_iter = 100000, C= 1, degree= 3, kernel='rbf')
rf=RandomForestClassifier(criterion= 'gini', max_depth= 5, max_features= 'auto', n_estimators=500)
knn=KNeighborsClassifier(n_neighbors= 2, p= 2, weights= 'uniform')

models=[lr,svm,rf,knn]
plt.figure(figsize=(10, 10))
for model in models:
    model.fit(X_train,y_train)
    lr_probs  = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    # Pintamos las curvas ROC
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model)
    # Etiquetas de los ejes
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title("ROC*CURVE Modelos")
pyplot.legend()
pyplot.show()