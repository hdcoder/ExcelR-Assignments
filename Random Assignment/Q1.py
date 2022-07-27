import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

Company_Data = pd.read_csv("Company_Data.csv")
print(Company_Data.head())
print()

print(Company_Data.shape)
print()

print(Company_Data.describe())
print()

print("Length of Sales Variable : ",len(Company_Data["Sales"]))
print("Mean of Sales Variable : ",Company_Data["Sales"].mean())
print(Company_Data["Sales"].sort_values())
print()

Company_Data["highsales"] = np.where((Company_Data["Sales"] < 9),"Low","High")
print(Company_Data)
print()

Company_Data["highsales"].describe()

Company_Data_new = Company_Data.iloc[:,1:]
print(Company_Data_new)
print()

Label_encoder = preprocessing.LabelEncoder()
Company_Data_new["ShelveLoc"]= Label_encoder.fit_transform(Company_Data_new["ShelveLoc"])
Company_Data_new["Urban"]= Label_encoder.fit_transform(Company_Data_new["Urban"])
Company_Data_new["US"]= Label_encoder.fit_transform(Company_Data_new["US"])
Company_Data_new["highsales"]= Label_encoder.fit_transform(Company_Data_new["highsales"])

print(Company_Data_new.head())
print()

Array = Company_Data_new.values
X = Array[:,:-1]
Y = Array[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)

model = RandomForestClassifier(n_estimators = 100, max_features = 4)

model.fit(X_train,Y_train)

predict = model.predict(X_train)

print(pd.Series(predict).value_counts())
print()

print(pd.crosstab(Y_train,predict))
print()

print(np.mean(predict == Y_train))
print()

predict = model.predict(X_test)

print(pd.Series(predict).value_counts())
print()

print(pd.crosstab(Y_test,predict))
print()

print(np.mean(predict == Y_test))
print()

importance = model.feature_importances_

print(importance)
print()

plt.figure(figsize=(15,10))

sns.barplot(X=["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"],Y = model.feature_importances_)
