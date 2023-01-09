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

Crime_Data = pd.read_csv("Crime_Data.csv")
print(Crime_Data.head())
print()

Crime_Data_1 = Crime_Data.copy()
Crime_Data_1.columns = ["City", "Murder", "Assault", "Urbanpop", "Rape"]
print(Crime_Data_1.head())
print()

Crime_Data_1.loc[:, "Total"] = Crime_Data_1.sum(numeric_only = True, axis = 1)
print(Crime_Data_1.head())
print()

print(Crime_Data_1.info())
print()

print(Crime_Data_1.describe())
print()

fig, ax = plt.subplots(figsize = (15, 13))
Stats = Crime_Data_1.sort_values('Total', ascending = False)

plt.figure()
sns.set_color_codes('dark')
sns.barplot(x = 'Total', y = 'City', data = Stats, label = 'Total', color = 'g')
sns.barplot(x = 'Assault', y= 'City', data = Stats, label='Assault', color ='b')
sns.barplot(x = 'Rape', y = 'City', data = Stats, label = 'Rape', color ='y')
sns.barplot(x = 'Murder', y = 'City', data = Stats, label = 'Murder', color = 'r')
ax.legend(ncol = 1, loc = 'lower right', frameon = True)
ax.set(xlim = (0,400), xlabel = 'No of atrrests forc each crime', ylabel = 'City')

plt.figure()
figure(figsize=(20, 10), dpi = 80)
plt.scatter(Crime_Data_1.City, Crime_Data_1.Murder, color = "r")
plt.scatter(Crime_Data_1.City, Crime_Data_1.Assault, color = "g")
plt.scatter(Crime_Data_1.City, Crime_Data_1.Rape, color = "b")
plt.scatter(Crime_Data_1.City, Crime_Data_1.Urbanpop, color = "y")
plt.xlabel("City-Name")
plt.ylabel("Rate")
plt.show()

X = Crime_Data_1[["Murder", "Assault", "Rape", "Urbanpop"]]

print(X)
print()

Crime_Data_1_norm = preprocessing.scale(X)
print(Crime_Data_1_norm)
print()

Crime_Data_1_norm = pd.DataFrame(Crime_Data_1_norm)
print(Crime_Data_1_norm)
print()

plt.figure(figsize=(10, 8))
WCSS = []

for i in range(1, 15):
    Kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 46)
    Kmeans.fit(Crime_Data_1_norm)
    WCSS.append(Kmeans.inertia_)
    
plt.figure()
plt.plot(range(1, 15), WCSS)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

Kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 46)
Y_kmeans = Kmeans.fit_predict(Crime_Data_1_norm)

Y_kmeans1  = Y_kmeans + 1
Cluster = list(Y_kmeans1)

Crime_Data_1['cluster'] = Cluster
print(Crime_Data_1)
print()

Kmeans_mean_cluster = pd.DataFrame(round(Crime_Data_1.groupby('cluster').mean(),1))
print(Kmeans_mean_cluster)
print()

plt.figure(figsize=(8,6))
sns.scatterplot(x = Crime_Data_1['Murder'], y = Crime_Data_1['Assault'], hue = Y_kmeans1)

plt.figure(figsize=(8,6))
sns.scatterplot(x = Crime_Data_1['Murder'], y = Crime_Data_1['Rape'], hue = Y_kmeans1)

plt.figure(figsize=(8,6))
sns.scatterplot(x = Crime_Data_1['Rape'], y = Crime_Data_1['Assault'], hue = Y_kmeans1)

Stats = Crime_Data_1.sort_values("Total", ascending=True)
Crime_Data_1_total= pd.DataFrame(Stats)
print(Crime_Data_1_total)
print()