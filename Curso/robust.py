#Regresor robusto para el caso en el que tenemos varios datos
#atipicos dentro de nuestro dataset
#En este ejemplo se enseña como usar un ciclo for para usar varios estimadores en pocas lineas
#de código
from numpy.core.fromnumeric import mean
import pandas as pd

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)

from sklearn.svm import SVR 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/whr2017.csv')
    print(dataset.head(5))

    X = dataset.drop(['country','score'], axis=1)
    y = dataset[['score']]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=42)

#usar variosestimadores al mismo tiempo con diccionarios
    estimadores = {
        'SVR':SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train,y_train)
        predictions = estimador.predict(X_test)
        print("="*64)
        print(name)
        print("MSE: ", mean_squared_error(y_test, predictions))