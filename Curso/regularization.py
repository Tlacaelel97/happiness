#Regularizar las features, aumentar un poco el sezgo para limitar la varianza

import pandas as pd
import sklearn
#modelos
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/whr2017.csv')
    print(dataset.describe())#descripcion estadistica

    X = dataset[['gdp','family', 'lifexp','freedom','corruption','generosity','dystopia']]
    y = dataset[['score']]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    #Regresores
    #definimos el modelo
    model_linear = LinearRegression().fit(X_train, y_train)
    #lo aplicamos
    y_predict_linear = model_linear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train,y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    #menor perdida es mejor, medicionde la perdida


    linear_loss= mean_squared_error(y_test, y_predict_linear)
    print("Liner loss: ", linear_loss)

    lasso_loss = mean_squared_error(y_test,y_predict_lasso)
    print("Lasso loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge loss: ", ridge_loss)

#los numeros mas grandes son los de mayor peso en el modelo

    print("="*32)
    print("Coef Lasso")
    print(modelLasso.coef_)

    print("="*32)
    print("Coef Ridge")
    print(modelRidge.coef_)

#Existe un punto medio llamado elasticnet, muy util para no perder info.