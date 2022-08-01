import numpy as np
import pandas as pd

import seaborn as sns

from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_decision_regions

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

Salary_train = pd.read_csv("Salary_Data_Train.csv")
Salary_test = pd.read_csv("Salary_Data_Test.csv")

print(Salary_train.info())
print()

LE = LabelEncoder()

Columns = ['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

for i in Columns:
    Salary_train[i] = LE.fit_transform(Salary_train[i])
    Salary_test[i] = LE.fit_transform(Salary_test[i])

Binary = LabelBinarizer()
Salary_train["Salary"] = Binary.fit_transform(Salary_train["Salary"])
Salary_test['Salary'] = Binary.fit_transform(Salary_test["Salary"])

print(Salary_train)
print()

print(Salary_test)
print()

Corr = Salary_train.corr()

plt.figure(figsize=(15,8))
sns.heatmap(Corr,annot=True)

plt.figure()
sns.countplot(data=Salary_test,x="occupation",hue="Salary")

plt.figure()
sns.distplot(Salary_train.Salary)

print(Salary_train["Salary"].value_counts())
print(Salary_test["Salary"].value_counts())
print()

X_train = Salary_train.iloc[:,0:13]
Y_train = Salary_train.iloc[:,13]
X_test = Salary_test.iloc[:,0:13]
Y_test = Salary_test.iloc[:,13]

X_train = norm_func(X_train)
X_test =  norm_func(X_test)

Model_linear = SVC(kernel = "linear")
Model_linear.fit(X_train,Y_train)
Pred_test_linear = Model_linear.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Pred_test_linear))
print()

Model_poly = SVC(kernel = "poly")
Model_poly.fit(X_train,Y_train)
Pred_test_poly = Model_poly.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Pred_test_poly))
print()

Model_rbf = SVC(kernel = "rbf")
Model_rbf.fit(X_train,Y_train)
Pred_test_rbf = Model_rbf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Pred_test_rbf))
print()

Model_sigmoid = SVC(kernel = "sigmoid")
Model_sigmoid.fit(X_train,Y_train)
Pred_test_sigmoid = Model_sigmoid.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Pred_test_sigmoid))