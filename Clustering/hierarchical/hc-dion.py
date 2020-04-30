#hierarchical clustering

#importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidean distances')
plt.show()

#fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualising the clusters
plt.scatter(X[y_hc == 0 ,0],X[y_hc == 0,1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(X[y_hc == 1 ,0],X[y_hc == 1,1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(X[y_hc == 2 ,0],X[y_hc == 2,1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(X[y_hc == 3 ,0],X[y_hc == 3,1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(X[y_hc == 4 ,0],X[y_hc == 4,1], s = 100, c = 'magenta', label = 'cluster 5')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()