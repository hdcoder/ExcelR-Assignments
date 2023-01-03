import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit 
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

Salary_Data = pd.read_csv("Salary_Data.csv")
print(Salary_Data.head())
print()

print(Salary_Data.info())
print()

plt.figure()
sns.displot(Salary_Data["Years Experience"])

plt.figure()
sns.displot(Salary_Data["Salary"])

print(Salary_Data.corr())
print()

plt.figure()
sns.regplot(x = Salary_Data["Years Experience"], y = Salary_Data["Salary"])

Model = smf.ols('Salary_Data["Salary"] ~ Salary_Data["Years Experience"]', data = Salary_Data["Salary"]).fit()

print(Model.params)
print()

print(Model.tvalues,Model.pvalues)
print()

print(Model.rsquared,Model.rsquared_adj)
print()

X = Salary_Data["Years Experience"]
Y = Salary_Data["Salary"]

b, m = polyfit(X, Y, 1)
plt.scatter(X, Y)
plt.plot(X, Y, '.')
plt.plot(X, b + m * X, '-')
plt.title('Salary Hike Scatter Plot')
plt.xlabel('Years Experience')
plt.ylabel('Salary')

print("Log transformation of X")
X_log = np.log(Salary_Data["Years Experience"])
Model = sm.OLS(Y,X).fit()
prediction = Model.predict(X_log)
print(Model.summary())
print()

print("Log transformation of Y")
Y_log = np.log(Salary_Data["Salary"])
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
X_sqrt = np.sqrt(Salary_Data["Years Experience"])
Model = sm.OLS(Y, X_sqrt).fit()
Predictions = Model.predict(X_sqrt)
print(Model.summary())
print()

print("Square Root transformation of Y")
Y_sqrt = np.sqrt(Salary_Data["Salary"])
Model = sm.OLS(Y_sqrt, X).fit()
Predictions = Model.predict(X)
print(Model.summary())
print()

print("Square Root transformation of X & Y")
Model = sm.OLS(Y_sqrt, X_sqrt).fit()
Predictions = Model.predict(X_sqrt)
print(Model.summary())
print()

print("Model with Square Root transformation of X has best adj-r-squared value.So have the best fit.")