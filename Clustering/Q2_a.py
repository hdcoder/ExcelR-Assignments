import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

Airlines =  pd.read_csv("EastWestAirlines.csv")

A = Airlines.iloc[:,[1,6]].values

dendro = dendrogram(linkage(A, method="ward"))

plt.title("Dendrogram Plot")
plt.ylabel("Euclidean Distances")
plt.xlabel("Customers")
plt.show()

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  

y_pred = hc.fit_predict(Airlines)

print(y_pred)
