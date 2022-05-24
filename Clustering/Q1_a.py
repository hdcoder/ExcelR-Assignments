import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

Crime =  pd.read_csv("Crime_Data.csv")

C = Crime.iloc[:,[1,6]].values

dendro = dendrogram(linkage(C, method="ward"))

plt.title("Dendrogram Plot")
plt.ylabel("Euclidean Distances")
plt.xlabel("Customers")
plt.show()

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  

y_pred = hc.fit_predict(Crime)

print(y_pred)