import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

import seaborn as sns

Company_Data = pd.read_csv("Company_Data.csv")
print(Company_Data.head())
print()

Label_Encoder = preprocessing.LabelEncoder()

Company_Data["ShelveLoc"] = Label_Encoder.fit_transform(Company_Data["ShelveLoc"])
Company_Data["Urban"] = Label_Encoder.fit_transform(Company_Data["Urban"])
Company_Data["US"] = Label_Encoder.fit_transform(Company_Data["US"])

print(Company_Data.head())
print()
print("Length of Sales Variable : ",len(Company_Data["Sales"]))
print()
print("Mean of Sales Variable : ",Company_Data["Sales"].mean())
print()
print(Company_Data["Sales"].sort_values())
print()

Company_Data["highsales"] = np.where((Company_Data["Sales"] < 9),"Low","High")
Company_Data["highsales"] = Label_Encoder.fit_transform(Company_Data["highsales"])
Company_Data_new = Company_Data.iloc[:,1:]
print(Company_Data_new.head())
print()

Array = Company_Data_new.values
X = Array[:,:-1]
Y = Array[:,-1]

print(X)
print(Y)
print()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

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

plt.figure(figsize=(15,10))
sns.barplot(X=["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"],y=Model.feature_importances_)