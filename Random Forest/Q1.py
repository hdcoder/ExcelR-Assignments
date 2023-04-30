import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

company_data = pd.read_csv("Company_Data.csv")

print(company_data.head)
print(company_data.shape)
print(company_data.describe)

print("Length of Sales Variable : ",len(company_data["Sales"]))
print("Mean of Sales Variable : ",company_data["Sales"].mean())

company_data["Sales"].sort_values()

company_data["highsales"] = np.where((company_data["Sales"] < 9),"Low","High")

print(company_data)

print(company_data["highsales"].describe())

company_data_new = company_data.iloc[:,1:]

print(company_data_new)

label_encoder = preprocessing.LabelEncoder()
company_data_new["ShelveLoc"]= label_encoder.fit_transform(company_data_new["ShelveLoc"])
company_data_new["Urban"]= label_encoder.fit_transform(company_data_new["Urban"])
company_data_new["US"]= label_encoder.fit_transform(company_data_new["US"])
company_data_new["highsales"]= label_encoder.fit_transform(company_data_new["highsales"])

print(company_data_new.head())

array = company_data_new.values
X = array[:,:-1]
y = array[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

model = RandomForestClassifier(n_estimators = 100, max_features = 4)
model.fit(X_train,y_train)

predict = model.predict(X_train)
print(pd.Series(predict).value_counts())

print(pd.crosstab(y_train,predict))

np.mean(predict == y_train)

predict = model.predict(X_test)
pd.Series(predict).value_counts()

np.mean(predict == y_test)

importance = model.feature_importances_
print(importance)

plt.figure(figsize=(15,10))
sns.barplot(x=["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"],y=model.feature_importances_)

