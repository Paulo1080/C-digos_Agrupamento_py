#!/usr/bin/python3

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
data = iris.data

km = KMeans(n_clusters=2).fit(data)

clusters = km.labels_

print('K-Means: Clusters Encontrados (Iris, K=2):')
print(clusters)


#####################################################################
### PLOT THE FOUND CLUSTERS
#####################################################################

import pandas
import matplotlib.pyplot as plt

df = pandas.DataFrame(iris.data)
df[4] = clusters

cluster0 = df[df[4] == 0]
cluster1 = df[df[4] == 1]

plt.scatter(cluster0.iloc[:,2] , cluster0.iloc[:,3], color='blue')
plt.scatter(cluster1.iloc[:,2] , cluster1.iloc[:,3], color='red')

plt.show()
