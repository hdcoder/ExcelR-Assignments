import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.stats.tests.test_influence

from numpy.polynomial.polynomial import polyfit

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

Bank_Data = pd.read_csv("Bank-full.csv", sep=";")
print(Bank_Data.head())
print()

Bank_Data_1 = Bank_Data.copy()
print(Bank_Data_1.head())
print()

print(Bank_Data_1.describe())
print()

print(Bank_Data_1.info())
print()

sns.pairplot(Bank_Data_1)

print(Bank_Data_1["y"].value_counts())
print()

No_sub = len(Bank_Data_1[Bank_Data_1["y"] == "no"])
Yes_sub = len(Bank_Data_1[Bank_Data_1["y"] == "yes"])
print(No_sub, Yes_sub)
print((Yes_sub/(Yes_sub + No_sub)) * 100)
print()

plt.figure()
pd.crosstab(Bank_Data_1["job"], Bank_Data_1["y"]).plot(kind = "bar")
plt.title("Crosstab")
plt.xlabel("Job")
plt.ylabel("Frequency of Subscription")

plt.figure()
Fig = pd.crosstab(Bank_Data_1["marital"], Bank_Data_1["y"])
Fig.div(Fig.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
plt.title("Marital VS Subscribed")
plt.xlabel("Marital")
plt.ylabel("Subscribed")

plt.figure()
Fig = pd.crosstab(Bank_Data_1.education, Bank_Data_1.y)
Fig.div(Fig.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Education  vs Subscribed ')
plt.xlabel('Education')
plt.ylabel('Profortion of Customers')

plt.figure()
Fig = pd.crosstab(Bank_Data_1.contact, Bank_Data_1.y)
Fig.div(Fig.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Contact  vs Subscribed ')
plt.xlabel('Contact')
plt.ylabel('Profortion of Customers')

plt.figure()
Fig = pd.crosstab(Bank_Data_1.poutcome, Bank_Data_1.y)
Fig.div(Fig.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.title('Poutcome  vs Subscribed ')
plt.xlabel('Poutcome')
plt.ylabel('Profortion of Customers')

plt.figure()
Bank_Data_1["age"].hist()
plt.title('Histogram for Age ')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.figure()
pd.crosstab(Bank_Data_1.month, Bank_Data_1.y).plot(kind='bar')
plt.title('Subscribed Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of subscribtion')

plt.figure()
Bank_Data_1.day.hist()
plt.title('Histogram of Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')

print(Bank_Data_1['housing'].value_counts())
print()

print(Bank_Data_1['loan'].value_counts())
print()

print(Bank_Data_1.isnull().sum())
print()

Bank_Data_1['default'] = Bank_Data_1['default'].map({'yes':1,'no':0})
Bank_Data_1['housing'] = Bank_Data_1['housing'].map({'yes':1,'no':0})
Bank_Data_1['loan'] = Bank_Data_1['loan'].map({'yes':1,'no':0})
Bank_Data_1['y'] = Bank_Data_1['y'].map({'yes':1,'no':0})

Bank_Data_1 = pd.get_dummies(Bank_Data_1,columns=['job'])
Bank_Data_1 = pd.get_dummies(Bank_Data_1,columns=['marital'])
Bank_Data_1 = pd.get_dummies(Bank_Data_1,columns=['education'])
Bank_Data_1 = pd.get_dummies(Bank_Data_1,columns=['month'])

print(Bank_Data_1.columns)
print()

Bank_Data_1 = Bank_Data_1.drop(['contact','poutcome'],axis=1)

print(Bank_Data_1.head())
print()

X = Bank_Data_1.loc[:, Bank_Data_1.columns != "y"]
Y = Bank_Data_1.loc[:, Bank_Data_1.columns == "y"]

Model = LogisticRegression()

RFE = RFE(Model,20)
RFE = RFE.fit(X, Y.values.ravel())

print(RFE.support_)
print(RFE.ranking_)
print()

X = Bank_Data_1[['default','housing','loan','job_housemaid','job_retired','job_student','marital_married','education_primary','education_unknown','month_aug','month_dec','month_feb','month_jan','month_jul','month_jun','month_mar','month_may','month_nov','month_oct','month_sep']]
Y = Bank_Data_1.loc[:, Bank_Data_1.columns == 'y']

Logit = sm.Logit(Y,X)
Result = Logit.fit()

print(Result.summary())
print()

print(Model.fit(X, Y))
print()

Pred_y = Model.predict(X)
print('Accuracy of logistic regression classfier on test set:{:.2f}'.format(Model.score(X,Y)))
print()

print(confusion_matrix(Y, Pred_y))
print()

print(classification_report(Y, Pred_y))
print()

print("Conclusion : Correct predictions - 39544 + 456 Incorrect predictions - 4833 + 467 Accuracy - 84%")