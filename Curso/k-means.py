"""
CLUSTERING
Aprendizaje no suupervisado, no conocemos las etiquetas, nosabemos nada y querenos descubrir como se agrupan
identificar valores atipicps.
Dos caso de aplicacion: sabemos cauntos grupos queremos k_means o spectral clusterin
si conocemos k, 
"""
#Asumimos que si sabemos cuantos grupos queremos
import pandas as pd

#si no tenemos alto poder de procesamiento
from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":

    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(10))

    X = dataset.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("Total de centros ", len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(X))

#modificar la dataset con los resultados
    dataset['group'] = kmeans.predict(X)
    print(dataset)