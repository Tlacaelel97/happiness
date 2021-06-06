"""
Validacion de modelos.
Todos  los modelos son malos, solo algunos son utiles.
Tipos: 
-Dividir datos en train y test, hold on- prueba rapida, principiante- baja capacidad de 
 computo
-Validacion cruzada(k-folds), forma plieges con diferentes partes del dataset
 para cubrir todo.- Definimos cuantas veces hacemos este procedimiento. recomendable en la mayoria 
 de los casos, si requiere la integracion de optimizacion paramétrica.
-LOOCV: Entrenamiento con todos los datos salvo 1, y asi con todos.- Requiere mucho poder,  
 se puede usar para pocos datos.
"""
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)

if __name__ == "__main__":

    dataset = pd.read_csv('./data/felicidad.csv')

    X = dataset.drop(['country','score'], axis =1)
    y = dataset['score']

    model = DecisionTreeRegressor()
    #cv define los plieges
    score = cross_val_score(model, X,y, cv=3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):#lo hara 3 veces
        print(train)
        print(test)

#de aqui se pueden usar los datos de train y test para el modelo