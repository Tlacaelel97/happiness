#cuando no conocemos k
import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(5))

    #no podemos entrenar una columna categorica
    X = dataset.drop('competitorname', axis = 1)

    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_))#verificacion de cuantos clusters decidio manejar
    print("="*70)
    print(meanshift.cluster_centers_)

    #integrarlo al dataset
    dataset['meanshift'] = meanshift.labels_
    print("="*70)
    print(dataset)
