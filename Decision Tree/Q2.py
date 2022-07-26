import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

import seaborn as sns

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

Fraud_Data = pd.read_csv("Fraud_Check.csv")
print(Fraud_Data.head())
print()

Label_Encoder = preprocessing.LabelEncoder()

Fraud_Data = pd.get_dummies(Fraud_Data, columns=["Undergrad","Marital.Status","Urban"],drop_first=True)

Fraud_Data["TaxInc"] = pd.cut(Fraud_Data["Taxable.Income"], bins=[10002,30000,99620], labels=["Risky","Good"])

print(Fraud_Data.head())
print()

Fraud_Data = pd.get_dummies(Fraud_Data, columns=["TaxInc"],drop_first=True)
print(Fraud_Data.tail())
print()

Fraud_Data_norm = norm_func(Fraud_Data.iloc[:,:])
print(Fraud_Data_norm.tail(10))

X = Fraud_Data.iloc[:,1:7]
Y = Fraud_Data["TaxInc_Good"]

print(X)
print(Y)
print()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

Model = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
Model.fit(X_train,Y_train)

Predict = Model.predict(X_test)
print(pd.Series(Predict).value_counts())
print()

pd.crosstab(Y_test,Predict)
print(np.mean(Predict == Y_test))
print()

Importance = Model.feature_importances_
print(Importance)
print()

tree.plot_tree(Model)

print(Y_train.value_counts())
print()

FN = ["City.Population","Work.Experience","Undergrad_YES","Marital.Status_Married","Marital.Status_Single","Urban_Yes"]
CN = ["Risky","Good"]
plt.fig()
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=300)
tree.plot_tree(Model,feature_names = FN, class_names=CN,filled = True)