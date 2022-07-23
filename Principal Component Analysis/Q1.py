import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

Wine_Data = pd.read_csv("Wine.csv")
print(Wine_Data.head())
print()

Wine_Data_1 = Wine_Data.iloc[:,1:]
print(Wine_Data_1.head())
print()

print(Wine_Data_1.info())
print()

print(Wine_Data_1.describe())
print()

Cor = Wine_Data_1.corr()
Cor.style.background_gradient(cmap='coolwarm')

Wine_Data_1_Norm = StandardScaler().fit_transform(Wine_Data_1)
print(Wine_Data_1_Norm)
print()

PCA = PCA(n_components = 13)

Principal_Components = PCA.fit_transform(Wine_Data_1_Norm)
print(Principal_Components)
print()

PC = range(1, PCA.n_components_+1)
plt.figure()
plt.bar(PC, PCA.explained_variance_ratio_, color='red')
plt.xlabel('Num of Principal Components')
plt.ylabel('Percentage of Variance')
plt.xticks(PC)

PCA_components = pd.DataFrame(Principal_Components)
print(PCA_components)
print()

plt.figure()
plt.scatter(PCA_components[0],PCA_components[1],alpha=.3,color='red')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

WCSS = []

for i in range(1,15):
    KMeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 24)
    KMeans.fit(PCA_components.iloc[:,:3])
    WCSS.append(KMeans.inertia_)

print(WCSS)
print()

plt.figure()
plt.plot(range(1,15),WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')

Model = KMeans(n_clusters = 3)
Model.fit(PCA_components.iloc[:,:2])
labels = Model.predict(PCA_components.iloc[:,:2])

plt.figure()
plt.scatter(PCA_components[0], PCA_components[1], c = labels)

K_new_df = pd.DataFrame(Principal_Components[:,0:2])
print(K_new_df.head())
print()

Model_k = KMeans(n_clusters=3)
Model_k.fit(K_new_df)
print(Model_k.labels_)
print()

Model_l = pd.Series(Model_k.labels_)

Wine_Data_1['clust'] = Model_l

print(K_new_df.head())
print()

print(Wine_Data_1.groupby(Wine_Data_1.clust).mean())
print()

Model_2 = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'ward')

H_cluster = Model_2.fit(PCA_components.iloc[:,:2])
Labels_2 = Model_2.labels_

X = PCA_components.iloc[:,:1]
Y = PCA_components.iloc[:,1:2]

plt.figure(figsize = (10, 7))  
plt.scatter(X, Y, c = Labels_2)

H_new_df = pd.DataFrame(Principal_Components[:,0:2])
print(H_new_df)
print()

HCF = linkage(H_new_df, method = "complete", metric = "euclidean")

plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')

sch.dendrogram(HCF, leaf_rotation=0, leaf_font_size=8)

H_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(H_new_df)

print(H_complete.labels_)
print()

Cluster_labels = pd.Series(H_complete.labels_)

Wine_Data_1['clust'] = Cluster_labels
print(Wine_Data_1.head())
print()

print(Wine_Data_1.groupby(Wine_Data_1.clust).mean())
print()