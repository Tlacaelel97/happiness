#----------------Metodos de ensamble-----------------------------------------------
"""
Bagging comparar varios estimadores y decide cual es la mejor
-Boosting: vamos pasando poco a poco a varios expertos, para llegar
a un concenso, viene de propulsar, toma clasificadores pequeños
y los itera para fortalecerlo. Al final usa un algoritmo de concenso para 
decidir la respuedta definitiva
"""
#clasificador
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    #cargar features
    X = dt_heart.drop(['target'],axis = 1) #inplace=true modifica el dataframe
    y = dt_heart['target']

    X_train, X_test, y_tran, y_test = train_test_split(X,y, test_size = 0.35)

    knn_class = KNeighborsClassifier().fit(X_train,y_tran)
    knn_pred = knn_class.predict(X_test)
    print("="*64)
    print(accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train,y_tran)
    bag_pred = bag_class.predict(X_test)
    print("="*64)
    print(accuracy_score(bag_pred, y_test))