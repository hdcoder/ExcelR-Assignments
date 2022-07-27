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

Fraud_Data = pd.read_csv("Fraud_check.csv")
print(Fraud_Data.head())
print()

Fraud_Data["income"]="<=30000"
Fraud_Data.loc[Fraud_Data["Taxable.Income"]>=30000,"income"]="Good"
Fraud_Data.loc[Fraud_Data["Taxable.Income"]<=30000,"income"]="Risky"

Fraud_Data.drop(["Taxable.Income"],axis=1,inplace=True)

Fraud_Data.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)
print(Fraud_Data.head())
print()

le = preprocessing.LabelEncoder()

for column_name in Fraud_Data.columns:
    if Fraud_Data[column_name].dtype == object:
        Fraud_Data[column_name] = le.fit_transform(Fraud_Data[column_name])
    else:
        pass
    
print(Fraud_Data.tail())
print()

Fraud_Data["income"].value_counts()

Features = Fraud_Data.iloc[:,0:5]
Labels = Fraud_Data.iloc[:,5]

Colnames = list(Fraud_Data.columns)
Predictors = Colnames[0:5]
Target = Colnames[5]

X_train,X_test,Y_train,Y_test = train_test_split(Features,Labels,test_size = 0.2,random_state=24)

X, Y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
model = RF(max_depth=2, random_state=0)
model.fit(X_train,Y_train)

Prediction = model.predict(X_train)

Accuracy = accuracy_score(Y_train,Prediction)
print(Accuracy)
print()

Confusion = confusion_matrix(Y_train,Prediction)
print(Confusion)
print()