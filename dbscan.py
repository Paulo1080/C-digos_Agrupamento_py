#!/usr/bin/python3

from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN

iris = load_iris()
data = iris.data

ds = DBSCAN(eps=0.9).fit(data)
clusters = ds.labels_

print('DBSCAN: Clusters Encontrados (Iris, Eps=0.9):')
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
