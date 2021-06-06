"""
OPTIMIZACION PARAMÉTRICA

tres enfoques
- Optimizacion manual: Escoher el modelo, documentacion para ver los parametros y ajustes,
 y probar a mano las configuraciones.
. Por grilla de parametros: Definir las variables a optimizar(diccionerios), identificar los
 posibles valores delos parametros
- Por busqueda Aleatorizada: No prueb atodo exhaustivamente si no que probara algunas
 configuraciones aleatorias.
"""

import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')

    print(dataset)

    X = dataset.drop(['country', 'rank', 'score'], axis =1)
    y = dataset[['score']]

    reg = RandomForestRegressor()

    #dfinir grilla de parametros
    parametros = {
        'n_estimators' : range(4,16),
        'criterion' : ['mse', 'mae'],
        'max_depth' : range(2,11)
    }
    #n_iter no sidce cuandas combinaciones hara
    rand_est = RandomizedSearchCV(reg,parametros, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    #automaticamente usa lo mejor
    print(rand_est.predict(X.loc[[0]]))