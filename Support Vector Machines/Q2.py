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

from metrics import confusion_matrix,accuracy_score

FF_Data = pd.read_csv("Forestfires.csv")

print(FF_Data)
print()

print(FF_Data.info())
print()

print(FF_Data.isnull().sum())
print()

Scale = StandardScaler()

FF_Data_1 = pd.get_dummies(FF_Data['size_category'],drop_first=True)
print(FF_Data_1.tail())
print()

FF_Data_2 = pd.concat([FF_Data, FF_Data_1], axis = 1)

X = FF_Data_1.iloc[:, 18:-1]
Y = FF_Data_2['small']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 15, test_size = 0.25)
Scale.fit(X_train)
X_train_std = Scale.transform(X_train)
X_test_std = Scale.transform(X_test)
X_train_df = pd.DataFrame(X_train_std, columns = X_train.columns)
X_test_df = pd.DataFrame(X_test_std, columns = X_test.columns)
print(X_train_df.head())
print()

SVM = SVC(C = 1.0, kernel = 'rbf', gamma = 0.50)
print(SVM)
print()

SVM.fit(X_train_df, Y_train)
SVM_indice = SVM.support_
SVM_class = SVM.n_support_
print(SVM_class)
print()
print(X_train)

X_train_arr = np.array(X_train)
print(X_train_arr)
print()

Y_pred = SVM.predict(X_test_df)
print(confusion_matrix(Y_test, Y_pred))
print()
print(accuracy_score(Y_test, Y_pred) * 100)
print()

SVM = SVC(C = 13, kernel = 'linear', gamma = 45, random_state = 5)
print(SVM)
print()

SVM.fit(X_train_df, Y_train)
Y_pred1 = SVM.predict(X_test_df)
print(Y_pred1)
print()

confusion_matrix(Y_test, Y_pred1)
print(Y_test)
print()
print(accuracy_score(Y_pred1, Y_test) * 100)
print()