#!/usr/bin/python3

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = load_iris()
data = iris.data

km = KMeans(n_clusters=2).fit(data)
clusters = km.labels_

ss = silhouette_score(data, clusters)

print('Largura da Silhueta (Iris, K=2):', ss)
