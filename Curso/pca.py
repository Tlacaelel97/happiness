from math import log
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')

    print(dt_heart.head(5))

    #quita la columna target
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    #normalizar los datos
    dt_features = StandardScaler().fit_transform(dt_features)

    #split de los datos
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    print(X_train.shape)
    print(y_train.shape)
    #configurar pca
    # n_components = min(n_muestras, n_features)
    pca= PCA(n_components=3)
    pca.fit(X_train)

    #no manda todo al mismo tiempo, para odenadores de bajo rendimiento
    ipca=IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    #medir la varianza, identificar los pesos importante
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_)
    plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    #aplicar pca
    dt_train = pca.transform(X_train)#aplicacion
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train,y_train)
    print("SCORE PCA: ", logistic.score(dt_test, y_test)) #conjuntode prueba contra los datos preparados para la prediccion


    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train,y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))