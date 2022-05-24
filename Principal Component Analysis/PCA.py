import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Wine = pd.read_csv("Wine.csv")

scalar = StandardScaler()
scalar.fit(Wine)
scaled_data = scalar.transform(Wine)

pca = PCA(n_components = 2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
x_pca.shape

plt.figure(figsize =(8, 6))

plt.scatter(x_pca[:, 0], x_pca[:, 1], cmap ='plasma')

plt.xlabel('First Principal Component')

plt.ylabel('Second Principal Component')

pca.components_

df_comp = pd.DataFrame(pca.components_)

plt.figure(figsize =(14, 6))

sns.heatmap(df_comp)

