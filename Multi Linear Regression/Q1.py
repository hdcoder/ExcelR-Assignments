import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor

def standard_values(vals):
    return(vals-vals.mean())/vals.std()

Startup_Data = pd.read_csv("50_Startups.csv")
print(Startup_Data.head(10))
print()

print(Startup_Data.info)
print()

print(Startup_Data.describe())
print()

print(Startup_Data.corr())
print()

sns.set_style(style="darkgrid")

sns.pairplot(data=Startup_Data)

Startup_Data_1 = Startup_Data.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'}, axis = 1)

Model = smf.ols('Profit ~ RDS + ADMS + MKTS', data=Startup_Data_1).fit()

print(Model.params)
print()

print(Model.tvalues,np.round(Model.pvalues,5))
print()

print(Model.rsquared,Model.rsquared_adj)
print()

SLR_A = smf.ols("Profit ~ ADMS", data = Startup_Data_1).fit()

print(SLR_A.tvalues, SLR_A.pvalues)
print()

SLR_M = smf.ols("Profit~MKTS", data=Startup_Data_1).fit()

print(SLR_M.tvalues, SLR_M.pvalues)
print()

SLR_AM = smf.ols("Profit~ADMS+MKTS", data=Startup_Data_1).fit()

print(SLR_AM.tvalues, SLR_AM.pvalues)
print()

RSQ_R = smf.ols('RDS~ADMS+MKTS',data = Startup_Data_1).fit().rsquared
VIF_R = 1/(1-RSQ_R)

RSQ_A = smf.ols('ADMS~RDS+MKTS',data = Startup_Data_1).fit().rsquared
VIF_A = 1/(1-RSQ_A)

RSQ_M = smf.ols('MKTS~RDS+ADMS',data = Startup_Data_1).fit().rsquared
VIF_M = 1/(1-RSQ_M)

D1 = {'Variables':['RDS','ADMS','MKTS'],'VIF':['VIF_R','VIF_M','VIF_A']}
VIF_DF = pd.DataFrame(D1)
print(VIF_DF)

plt.figure()
sm.qqplot(Model.resid, line="q")
plt.title("Normal Q-Q plot for residuals")
plt.show()

plt.figure()
plt.scatter(standard_values(Model.fittedvalues),standard_values(Model.resid))
plt.title('Residual Plot')
plt.xlabel('standarized fitted value')
plt.ylabel('standarized residual values')
plt.show()

fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(Model,'RDS',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(Model,'ADMS',fig=fig)
plt.show()

fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(Model,'MKTS',fig=fig)
plt.show()

(c,_) = Model.get_influence().cooks_distance
print(c)
print()

fig = plt.figure(figsize=(20,7))
plt.stem(np.arange(len(Startup_Data_1)),np.round(c,5))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance ')
plt.show()

print(np.argmax(c),np.max(c))
print()

influence_plot(Model)
plt.show()

k = Startup_Data_1.shape[1]
n = Startup_Data_1.shape[0]
leverage_cutoff = (3*(k+1))/n

print(leverage_cutoff)
print()

Startup_Data_1[Startup_Data_1.index.isin([49])]

Startup_Data_2 = Startup_Data_1.drop(Startup_Data_1.index[[49]],axis=0).reset_index(drop=True)

print(Startup_Data_2)
print()

while np.max(c)>0.5:
    Model = smf.ols('Profit ~ RDS+ADMS+MKTS',data = Startup_Data_2).fit()
    (c,_) = Model.get_influence().cooks_distance
    np.argmax(c),np.max(c)
    Startup_Data_2 = Startup_Data_1.drop(Startup_Data_1.index[[49]],axis=0).reset_index(drop=True)

print(Startup_Data_2)
print()

Final_model = smf.ols('Profit ~ RDS+ADMS+MKTS',data = Startup_Data_2).fit()

print(Final_model.rsquared,Final_model.aic)
print()

print('Thus model accuracy is improved to : ',Final_model.rsquared)
print()

New_Data = pd.DataFrame({'RDS':70000,'ADMS':90000,'MKTS':140000},index=[0])

print(New_Data)
print()

print(Final_model.predict(New_Data))
print()

Pred_y = Final_model.predict(Startup_Data_2)

print(Pred_y)
print()

D2 = {'Prep_Model':['Model','Final_Model'],'Rsquared':[Model.rsquared,Final_model.rsquared]}
Table = pd.DataFrame(D2)

print(Table)
print()