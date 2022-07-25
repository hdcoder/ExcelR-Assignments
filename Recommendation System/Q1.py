import pandas as pd 
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation

Books_data = pd.read_csv("Book_2.csv", encoding= "ISO-8859-1")
print(Books_data.head())
print()

Books_data_1 = Books_data.iloc[:, 1:]
print(Books_data_1.head())
print()

print(Books_data_1.sort_values(["User.ID"]))
print()

print(len(Books_data_1["User.ID"].unique()))
print()

print(len(Books_data_1["Book.Title"].unique()))
print()

Books_data_2 = Books_data_1.pivot_table(index = "User.ID", columns = "Book.Title", values = "Book.Rating").reset_index(drop = True)
print(Books_data_2)
print()

Books_data_2.index = Books_data_1["User.ID"].unique()
print(Books_data_2)
print()

Books_data_2.fillna(0, inplace = True)
print(Books_data_2)
print()

User_1 = 1 - pairwise_distances(Books_data_2, metric = "cosine")
print(User_1)
print()

User_2 = pd.DataFrame(User_1)
print(User_2)
print()

User_2.index = Books_data_1['User.ID'].unique()
User_2.columns = Books_data_1['User.ID'].unique()
print(User_2)
print()
print(User_2.idxmax(axis = 1))
print()

print(Books_data_1[(Books_data_1["User.ID"] == 162107) | (Books_data_1["User.ID"] == 276726) ])
print()

print(Books_data_1[(Books_data_1["User.ID"] == 276729) | (Books_data_1["User.ID"] == 276726) ])