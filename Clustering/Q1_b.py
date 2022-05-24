import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

Crime =  pd.read_csv("Crime_Data.csv")

C = Crime.iloc[:,[1,6]].values

wcss_list= []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(A)
    wcss_list.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)

y_predict= kmeans.fit_predict(C)

plt.scatter(C[y_predict == 0, 0], C[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(C[y_predict == 1, 0], C[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(C[y_predict== 2, 0], C[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
plt.scatter(C[y_predict == 3, 0], C[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(C[y_predict == 4, 0], C[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')

plt.title('Clusters of Crime Data')
plt.xlabel('Balance')
plt.ylabel('Bonus_miles')
plt.legend()
plt.show()