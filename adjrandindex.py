#!/usr/bin/python3

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

iris = load_iris()
data = iris.data
classes = iris.target

km = KMeans(n_clusters=3).fit(data)
clusters = km.labels_

ari = adjusted_rand_score(classes, clusters)

print('Indice de Rand Ajustado (Iris, K=3):', ari)
