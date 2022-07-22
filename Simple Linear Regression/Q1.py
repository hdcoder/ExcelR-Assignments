import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit 
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

Delivery_Data = pd.read_csv("Delivery_Time.csv")
print(Delivery_Data.head())
print()

print(Delivery_Data.info())
print()

plt.figure()
sns.distplot(Delivery_Data["Delivery Time"])

plt.figure()
sns.distplot(Delivery_Data["Sorting Time"])

print(Delivery_Data.corr())
print()

plt.figure()
sns.regplot(x = Delivery_Data["Delivery Time"], y = Delivery_Data["Sorting Time"])

Model = smf.ols('Delivery_Data["Delivery Time"] ~ Delivery_Data["Sorting Time"]', data = Delivery_Data["Delivery Time"]).fit()

print(Model.params)
print()

print(Model.tvalues,Model.pvalues)
print()

print(Model.rsquared,Model.rsquared_adj)
print()

New_Data = pd.Series([11,20])
print(New_Data)
print()

Data_Pred = pd.DataFrame(New_Data,columns = ["Sorting Time"])
print(Data_Pred)
print(Model.predict(Data_Pred))
print()

Y = Delivery_Data["Delivery Time"]
X = Delivery_Data["Sorting Time"]

print("Log transformation of X")
X_log = np.log(Delivery_Data["Sorting Time"])
Model = sm.OLS(Y,X).fit()
Prediction = Model.predict(X_log)
print(Model.summary())
print()

print("Log transformation of Y")
Y_log = np.log(Delivery_Data["Delivery Time"])
Model = sm.OLS(Y_log, X).fit()
Predictions = Model.predict(X)
print(Model.summary())
print()

print("Log transformation of X & Y")
Model = sm.OLS(Y_log, X_log).fit()
Predictions = Model.predict(X_log)
print(Model.summary())
print()

print("Square Root transformation of X")
X_sqrt = np.sqrt(Delivery_Data["Sorting Time"])
Model = sm.OLS(Y, X_sqrt).fit()
Predictions = Model.predict(X_sqrt)
print(Model.summary())
print()

print("Square Root transformation of Y")
Y_sqrt = np.sqrt(Delivery_Data["Delivery Time"])
Model = sm.OLS(Y_sqrt, X).fit()
Predictions = Model.predict(X)
print(Model.summary())
print()

print("Square Root transformation of X & Y")
Model = sm.OLS(Y_sqrt, X_sqrt).fit()
Predictions = Model.predict(X_sqrt)
print(Model.summary())
print()

print("Model in which Square Root of X and Y is taken gives best Adj-R squared value.This model should be accepted.")