import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

from scipy.special import comb
from itertools import combinations, permutations

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from mpl_toolkits.mplot3d import Axes3D

def gen_rules(df,confidence,support,Book_Data_1):
    AP = {}
    for i in confidence:
        AP_i = apriori(Book_Data_1, support, True)
        Rule = association_rules(AP_i, min_threshold = i)
        AP[i] = len(Rule.antecedents)
    return pd.Series(AP).to_frame("Support: %s"%support)

Book_Data = pd.read_csv("Book.csv")

print(Book_Data.head())
print()

Book_Data_1 = Book_Data.copy()

print(Book_Data_1.info())
print()

print(Book_Data_1.describe())
print()

print(Book_Data_1.isnull().sum())
print()

Item_Sets = {}
Transaction_Enc = TransactionEncoder()
Transaction_Enc_Ary = Transaction_Enc.fit(Book_Data_1).transform(Book_Data_1)
AP = pd.DataFrame(Transaction_Enc_Ary, columns = Transaction_Enc.columns_)

AP.sum().to_frame('Frequency').sort_values('Frequency',ascending=False)[:25].plot(kind='bar',figsize=(10,8),title="Frequent Items")

AP_0_5 = {}
AP_1 = {}
AP_5 = {}
AP_1_0 = {}

Confidence = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

Confs = []

for i in [0.001,0.005,0.01,0.05,0.1]:
    AP_i = gen_rules(AP,Confidence,i,Book_Data_1)
    Confs.append(AP_i)
    
All_Conf = pd.concat(Confs,axis = 1)
All_Conf.plot(figsize=(12,14),grid=True)
plt.ylabel('Rules')
plt.xlabel('Confidence')
plt.show()

AP_final =  apriori(AP,0.001,True)
Rules_final = association_rules(AP_final, min_threshold = .4, support_only = False)
Rules_final[Rules_final['confidence'] > 0.5]

Support = Rules_final["support"]
Confidence =  Rules_final["confidence"]
Lift = Rules_final["lift"]

Fig1 = plt.figure()
Ax1 = Fig1.add_subplot(111, projection = "3d")
Ax1.scatter(Support,Confidence,Lift)
Ax1.set_xlabel("Support")
Ax1.set_ylabel("Confidence")
Ax1.set_zlabel("Lift")

Fig2 = plt.figure()
plt.scatter(Support,Confidence,c = Lift,cmap = 'gray')
plt.colorbar()
plt.xlabel("Support")
plt.ylabel("Confidence")