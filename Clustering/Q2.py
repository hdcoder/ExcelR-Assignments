import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import scipy.cluster.hierarchy as sch 

from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering

Airlines_Data = pd.read_csv("East_West_Airlines.csv")
Airlines_Data_1 = Airlines_Data.copy()

print(Airlines_Data_1.head())
print()

Airlines_Data_1_norm = preprocessing.scale(Airlines_Data_1)
print(Airlines_Data_1_norm)
print()

Airlines_Data_1_norm = pd.DataFrame(Airlines_Data_1_norm)
print(Airlines_Data_1_norm)
print()

plt.figure(figsize=(10,8))
WCSS=[]

for i in range(1,15):
    KMeans = KMeans(n_clusters=i,init='k-means++',random_state=64)
    KMeans.fit(Airlines_Data_1_norm)
    WCSS.append(KMeans.inertia_)

plt.figure()
plt.plot(range(1,15),WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

Dendrogram = sch.dendrogram(sch.linkage(Airlines_Data_1_norm, method = 'ward'))

X = Airlines_Data_1_norm.values

Model = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

H_cluster = Model.fit(X)
Labels = Model.labels_

plt.figure()
plt.scatter(X[Labels==0,0],X[Labels==0,1],s=50,marker='o',color='red')
plt.scatter(X[Labels==1,0],X[Labels==1,1],s=50,marker='o',color='blue')
plt.scatter(X[Labels==2,0],X[Labels==2,1],s=50,marker='o',color='green')

KMeans = KMeans(n_clusters = 3, init = 'k-means++', random_state=42)
KMeans = KMeans.fit_predict(Airlines_Data_1_norm)
print(KMeans)
print()

KMeans1 = KMeans + 1
KCluster = list(KMeans1)

Airlines_Data_1['k_cluster'] = KCluster
print(Airlines_Data_1)
print()

KMeans_Mean_Cluster = pd.DataFrame(round(Airlines_Data_1.groupby('k_cluster').mean(),1))
print(KMeans_Mean_Cluster)
print()

pd.DataFrame(round(Airlines_Data_1.groupby('k_cluster').count(),1))

plt.figure()
plt.scatter(X[:,0],X[:,1],c=KMeans,s=50,cmap='viridis')