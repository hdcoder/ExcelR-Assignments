import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor

Toyota = pd.read_csv("Toyota.csv", encoding = "latin1")

print(Toyota.head())
print()

print(Toyota.info())
print()

Toyota_1 = pd.concat([Toyota.iloc[:,2:4], Toyota.iloc[:,6:7], Toyota.iloc[:,8:9], Toyota.iloc[:,12:14], Toyota.iloc[:,15:18]],axis=1)
print(Toyota_1)
print()

Toyota_2 = Toyota_1.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
print(Toyota_2)
print()

Toyota_3 = Toyota_2.drop_duplicates().reset_index(drop = True)
print(Toyota_3)
print()

print(Toyota_3.describe())
print()

print(Toyota_3.corr())
print()

sns.set_style(style = "darkgrid")

sns.pairplot(Toyota_3)

Model = smf.ols('Price ~ Age + KM + HP + CC + Doors + Gears + QT + Weight',data = Toyota_3).fit()

print(Model.params)
print()

print(Model.tvalues,np.round(Model.pvalues,5))
print()

print(Model.rsquared,Model.rsquared_adj)
print()

X = Toyota_3[['Age','KM','HP','CC','Doors','Gears','QT','Weight']]
Y = Toyota_3[['Price']]

print(Model.summary())
print()

Infl = Model.get_influence()
Summ_df = Infl.summary_frame()
Summ_df.sort_values( 'cooks_d', ascending= False)

Infl.plot_influence()

VIF = pd.DataFrame()
VIF["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
VIF["features"] = X.columns

VIF.round(1)

X1 = Toyota_3[['Age','KM','HP','CC','Doors','Gears','QT']]
Model_1 = sm.OLS(Y, X1).fit()

print(Model_1.summary())
print()

Toyota_4 = Toyota_3.drop(Toyota_3.index[80])

X1 = Toyota_4[['Age','KM','HP','CC','Doors','Gears','QT']]
Y1 = Toyota_4[['Price']]

Model_2 = sm.OLS(Y1, X1).fit()

Predictions2 = Model_2.predict(X1)

print(Model_2.summary())
print()